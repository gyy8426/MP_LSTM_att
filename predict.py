from utils import *
import copy


class Predict(object):

    def build_sampler(self, layers, tparams, options, use_noise, trng):
        # context: #annotations x dim
        ctx0 = T.matrix('ctx_sampler', dtype='float32')
        ctx_mask = T.vector('ctx_mask', dtype='float32')

        ctx_ = ctx0
        counts = ctx_mask.sum(-1)
        ctx_mean = ctx_.sum(0)/counts

        # initial state/cell
        tu_init_state = [T.alloc(0., options['tu_dim'])]
        tu_init_memory = [T.alloc(0., options['tu_dim'])]
        mu_init_state = [T.alloc(0., options['mu_dim'])]
        mu_init_memory = [T.alloc(0., options['mu_dim'])]
        if options['init_tulstm']:
            tu_init_state = [layers.get_layer('ff')[1](
                tparams, ctx_mean, prefix='ff_state', activ='tanh')]
            tu_init_memory = [layers.get_layer('ff')[1](
                tparams, ctx_mean, prefix='ff_memory', activ='tanh')]

        print 'Building f_init...',
        f_init = theano.function([ctx0, ctx_mask], [ctx0]+tu_init_state+tu_init_memory+
                                 mu_init_state+mu_init_memory, name='f_init',
                                 on_unused_input='ignore',
                                 profile=False)
        print 'Done'

        x = T.vector('x_sampler', dtype='int64')

        tu_init_state = [T.matrix('tu_init_state', dtype='float32')]
        tu_init_memory = [T.matrix('tu_init_memory', dtype='float32')]
        mu_init_state = [T.matrix('mu_init_state', dtype='float32')]
        mu_init_memory = [T.matrix('mu_init_memory', dtype='float32')]

        # if it's the first word, emb should be all zero
        emb = T.switch(x[:, None] < 0,
                       T.alloc(0., 1, tparams['Wemb'].shape[1]), tparams['Wemb'][x])
        tu_lstm = layers.get_layer('lstm')[1](tparams, emb, one_step=True,
                                              init_state=tu_init_state[0],
                                              init_memory=tu_init_memory[0],
                                              prefix='tu_lstm')
        mu_lstm = layers.get_layer(options['att_type'])[1](options, tparams, tu_lstm[0],
                                                   mask=None, context=ctx_[None,:,:],ctx_mean=ctx_mean,
                                                   one_step=True,
                                                   init_state=mu_init_state[0],
                                                   init_memory=mu_init_memory[0],
                                                   trng=trng,
                                                   use_noise=use_noise,
                                                   prefix='mu_lstm')
        tu_next_state = [tu_lstm[0]]
        tu_next_memory = [tu_lstm[1]]
        mu_next_state = [mu_lstm[0]]
        mu_next_memory = [mu_lstm[1]]

        proj_h = mu_lstm[0]
        
        alphas = mu_lstm[2]
        ctxs = mu_lstm[3]
        if options['use_dropout']:
            proj_h = layers.dropout_layer(proj_h, use_noise, trng)
        # compute word probabilities
        logit = layers.get_layer('ff')[1](
            tparams, proj_h, prefix='ff_logit_lstm', activ='linear')
        if options['prev2out']:
            logit += emb
        if options['ctx2out']:
            logit += layers.get_layer('ff')[1](
                tparams, ctxs, prefix='ff_logit_ctx', activ='linear')
        logit = tanh(logit)
        if options['use_dropout']:
            logit = layers.dropout_layer(logit, use_noise, trng)

        logit = layers.get_layer('ff')[1](
            tparams, logit, prefix='ff_logit', activ='linear')
        # logit_shp = logit.shape
        next_probs = T.nnet.softmax(logit)
        next_sample = trng.multinomial(pvals=next_probs).argmax(1)

        # next word probability
        print 'building f_next...'
        f_next = theano.function([x, ctx0, ctx_mask]+
                                 tu_init_state+tu_init_memory+
                                 mu_init_state+mu_init_memory,
                                 [next_probs, next_sample]+
                                 tu_next_state+tu_next_memory+
                                 mu_next_state+mu_next_memory,
                                 name='f_next', profile=False,
                                 on_unused_input='ignore')
        print 'Done'
        return f_init, f_next

    def gen_sample(self, tparams, f_init, f_next, ctx0, ctx_mask,
                   trng=None, k=1, maxlen=30, stochastic=False):
        '''
        ctx0: (26,1024)
        ctx_mask: (26,)
        '''

        if k > 1:
            assert not stochastic, 'Beam search does not support stochastic sampling'

        sample = []
        sample_score = []
        if stochastic:
            sample_score = 0

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype('float32')

        # [(26,1024),(512,),(512,)]
        rval = f_init(ctx0, ctx_mask)
        ctx0 = rval[0]

        # next lstm and stacked lstm state and memory
        next_states = []
        next_memorys = []
        n_layers_lstm = 2
        for lidx in xrange(n_layers_lstm):
            next_states.append([])
            next_memorys.append([])
            next_states[lidx].append(rval[2*lidx+1])
            next_states[lidx][-1] = next_states[lidx][-1].reshape([live_k, next_states[lidx][-1].shape[0]])
            next_memorys[lidx].append(rval[2*lidx+2])
            next_memorys[lidx][-1] = next_memorys[lidx][-1].reshape([live_k, next_memorys[lidx][-1].shape[0]])

        next_w = -1 * np.ones((1,)).astype('int64')
        # next_state: [(1,512)]
        # next_memory: [(1,512)]
        for ii in xrange(maxlen):
            # return [(1, 50000), (1,), (1, 512), (1, 512)]
            # next_w: vector
            # ctx: matrix
            # ctx_mask: vector
            # next_state: [matrix]
            # next_memory: [matrix]
            rval = f_next(*([next_w, ctx0, ctx_mask] +
                            next_states[0] + next_memorys[0] +
                            next_states[1] + next_memorys[1]))
            next_p = rval[0]
            next_w = rval[1] # already argmax sorted

            next_states = []
            next_memorys = []
            for lidx in xrange(n_layers_lstm):
                next_states.append([])
                next_memorys.append([])
                next_states[lidx].append(rval[2*lidx+2])
                next_memorys[lidx].append(rval[2*lidx+3])

            if stochastic:
                sample.append(next_w[0]) # take the most likely one
                sample_score += next_p[0,next_w[0]]
                if next_w[0] == 0:
                    break
            else:
                # the first run is (1,50000)
                cand_scores = hyp_scores[:,None] - np.log(next_p)
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:(k-dead_k)]

                voc_size = next_p.shape[1]
                trans_indices = ranks_flat / voc_size # index of row
                word_indices = ranks_flat % voc_size # index of col
                costs = cand_flat[ranks_flat]

                new_hyp_samples = []
                new_hyp_scores = np.zeros(k-dead_k).astype('float32')

                new_hyp_states = []
                new_hyp_memories = []
                for lidx in xrange(n_layers_lstm):
                    new_hyp_states.append([])
                    new_hyp_memories.append([])
                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(hyp_samples[ti]+[wi])
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    for lidx in np.arange(n_layers_lstm):
                        new_hyp_states[lidx].append(copy.copy(next_states[lidx][0][ti]))
                        new_hyp_memories[lidx].append(copy.copy(next_memorys[lidx][0][ti]))

                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_states = []
                hyp_memories = []
                for lidx in xrange(n_layers_lstm):
                    hyp_states.append([])
                    hyp_memories.append([])

                for idx in xrange(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == 0:
                        sample.append(new_hyp_samples[idx])
                        sample_score.append(new_hyp_scores[idx])
                        dead_k += 1
                    else:
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        for lidx in xrange(n_layers_lstm):
                            hyp_states[lidx].append(new_hyp_states[lidx][idx])
                            hyp_memories[lidx].append(new_hyp_memories[lidx][idx])
                hyp_scores = np.array(hyp_scores)
                live_k = new_live_k

                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break

                next_w = np.array([w[-1] for w in hyp_samples])
                next_states = []
                next_memorys = []
                for lidx in xrange(n_layers_lstm):
                    next_states.append([])
                    next_memorys.append([])
                    next_states[lidx].append(np.array(hyp_states[lidx]))
                    next_memorys[lidx].append(np.array(hyp_memories[lidx]))

        if not stochastic:
            # dump every remaining one
            if live_k > 0:
                for idx in xrange(live_k):
                    sample.append(hyp_samples[idx])
                    sample_score.append(hyp_scores[idx])

        return sample, sample_score, next_states, next_memorys

    def sample_execute(self, engine, options, tparams, f_init, f_next, x, ctx, mask_ctx, trng):
        stochastic = False
        for jj in xrange(np.minimum(10, x.shape[1])):
            sample, score, _, _ = self.gen_sample(tparams, f_init, f_next, ctx[jj], mask_ctx[jj],
                                                  trng=trng, k=5, maxlen=30, stochastic=stochastic)
            if not stochastic:
                best_one = np.argmin(score)
                sample = sample[best_one]
            else:
                sample = sample
            print 'Truth ', jj, ': ',
            for vv in x[:, jj]:
                if vv == 0:
                    break
                if vv in engine.ix_word:
                    print engine.ix_word[vv],
                else:
                    print 'UNK',
            print
            for kk, ss in enumerate([sample]):
                print 'Sample (', jj, ') ', ': ',
                for vv in ss:
                    if vv == 0:
                        break
                    if vv in engine.ix_word:
                        print engine.ix_word[vv],
                    else:
                        print 'UNK',
            print
