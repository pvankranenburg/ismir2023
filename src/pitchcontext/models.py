"""Models using the weighted pitch context vector."""

import numpy as np
from numpy.linalg import norm
from numpy import inf
from matplotlib import pyplot as plt
from copy import deepcopy

from .pitchcontext import PitchContext
from .song import Song
from .base40 import base40

#from datetime import datetime
#print(__file__, datetime.now().strftime("%H:%M:%S"))

#distances
def cosineSim(v1, v2):
    """Cosine Similarity of `v1` and `v2`, both 1D numpy arrays"""
    return np.dot(v1,v2)/(norm(v1)*norm(v2))

def normalizedCosineSim(v1, v2):
    """Cosine Similarity of `v1` and `v2`, both 1D numpy arrays. Scaled between 0 and 1."""
    return (1.0 + np.dot(v1,v2)/(norm(v1)*norm(v2))) / 2.0

def normalizedCosineDist(v1, v2):
    """One minus the normalized cosine similarity"""
    return 1.0-normalizedCosineSim(v1, v2)

def sseDist(v1, v2):
    """Sum of squared differences between `v1` and `v2`, both 1D numpy arrays."""
    return np.sum((v1-v2)**2)


#find out how dissonant a note is in its context
#consonant in base40:
# perfect prime : Dp = 0
# minor third : Dp = 11
# major third : Dp = 12
# perfect fourth : Dp = 17
# perfect fifth : Dp = 23
# minor sixth : Dp = 28
# major sixth : Dp = 29

def computeDissonance(
    song : Song, 
    wpc : PitchContext,
    combiner=np.maximum,
    normalizecontexts = False,
    consonants40 = [0, 11, 12, 17, 23, 28, 29]
):
    """
    Computes for each note the dissonance of the note given its context.

    Parameters
    ----------
    song : Song
        An instance of the Song class.
    wpd : WeightedPitchContext
        An instance of the WeightedPitchContext class, containing a weighted pitch context vector for each note.
    combiner : function of two 1D numpy arrays, default=numpy.maximum
        Combines the dissonance of preceding context and dissonance of following context in one value.
        Default: take the maximum.
    normalizecontexts : bool, default=False
        Normalize (sum-1.0) the context vectors before computing dissonance
    consonants : list of ints
        Intervals in base40 pitch encoding that are considered consonant.
    
    Returns
    -------
    dissonance_pre, dissonance_post, dissonance_context : numpy 1D arrays
        with a dissonance level for each note, respective the dissonance within the preceding context, the
        dissonance within the following context, and the dissonance within the full context.
    """
    song_length = len(wpc.ixs)

    dissonants = np.ones( (40,) )
    dissonants[consonants40] = 0.0

    #store result
    dissonance_pre = np.zeros( song_length )
    dissonance_post = np.zeros( song_length )
    dissonance_context = np.zeros( song_length )

    for ix, context in enumerate(wpc.pitchcontext): #go over the notes...

        pitch40 = song.mtcsong['features']['pitch40'][wpc.ixs[ix]]-1

        #make copy of context
        context = np.copy(context)

        #normalize contexts: sum of context is 1.0 (zero stays zero)
        if normalizecontexts:
            if np.sum(context[:40]) > 0.0:
                context[:40] = context[:40] / np.sum(context[:40])
            if np.sum(context[40:]) > 0.0:
                context[40:] = context[40:] / np.sum(context[40:])


        intervals_pre  = np.roll(context[:40], -pitch40)
        intervals_post = np.roll(context[40:], -pitch40)

        dissonance_pre[ix] = np.sum(np.multiply(intervals_pre, dissonants))
        dissonance_post[ix] = np.sum(np.multiply(intervals_post, dissonants))

        #if context is empty: value should be np.nan
        if len(wpc.contexts_pre[ix]) == 0:
            dissonance_pre[ix] = np.nan
        if len(wpc.contexts_post[ix]) == 0:
            dissonance_post[ix] = np.nan

    #combine pre and post context
    dissonance_context = combiner(dissonance_pre, dissonance_post)

    return dissonance_pre, dissonance_post, dissonance_context


def computeConsonance(
    song : Song, 
    wpc : PitchContext,
    combiner=np.minimum,
    normalizecontexts = False,
    consonants40 = [0, 11, 12, 17, 23, 28, 29]
):
    """
    Computes for each note the consonance of the note given its context.

    Parameters
    ----------
    song : Song
        An instance of the Song class.
    wpd : WeightedPitchContext
        An instance of the WeightedPitchContext class, containing a weighted pitch context vector for each note.
    combiner : function of two 1D numpy arrays, default=numpy.maximum
        Combines the consonance of preceding context and consonance of following context in one value.
        Default: take the maximum.
    normalizecontexts : bool, default=False
        Normalize (sum-1.0) the context vectors before computing consonance
    consonants : list of ints
        Intervals in base40 pitch encoding that are considered consonant.
    
    Returns
    -------
    consonance_pre, consonance_post, consonance_context : numpy 1D arrays
        with a consonance level for each note, respective the consonance within the preceding context, the
        consonance within the following context, and the consonance within the full context.
    """
    song_length = len(wpc.ixs)

    consonants = np.zeros( (40,) )
    consonants[consonants40] = 1.0

    #store result
    consonance_pre = np.zeros( song_length )
    consonance_post = np.zeros( song_length )
    consonance_context = np.zeros( song_length )

    for ix, context in enumerate(wpc.pitchcontext): #go over the notes...

        pitch40 = song.mtcsong['features']['pitch40'][wpc.ixs[ix]]-1

        #make copy of context
        context = np.copy(context)

        #normalize contexts: sum of context is 1.0 (zero stays zero)
        if normalizecontexts:
            if np.sum(context[:40]) > 0.0:
                context[:40] = context[:40] / np.sum(context[:40])
            if np.sum(context[40:]) > 0.0:
                context[40:] = context[40:] / np.sum(context[40:])

        intervals_pre  = np.roll(context[:40], -pitch40)
        intervals_post = np.roll(context[40:], -pitch40)

        consonance_pre[ix] = np.sum(np.multiply(intervals_pre, consonants))
        consonance_post[ix] = np.sum(np.multiply(intervals_post, consonants))

        #if context is empty: value should be np.nan
        if len(wpc.contexts_pre[ix]) == 0:
            consonance_pre[ix] = np.nan
        if len(wpc.contexts_post[ix]) == 0:
            consonance_post[ix] = np.nan

    #combine pre and post context
    consonance_context = combiner(consonance_pre, consonance_post)

    return consonance_pre, consonance_post, consonance_context


def computePrePostDistance(
    song : Song,
    wpc : PitchContext,
    vectorDist=normalizedCosineDist
):
    """Computes for each note the distance between the preceding and the following context.

    Parameters
    ----------
    song : Song
        Ojbect with song data.
    wpc : PitchContext
        Object with pitch context data
    vectorDist : function, default=normalizedCosineDist
        Function to compute the distance between two 1D vectors

    Returns
    -------
    numpy array
        1D numpy array with a distance value for each note.
    """
    res = np.zeros( len(wpc.pitchcontext) )
    for ix in range(len(wpc.pitchcontext)):
        res[ix] = vectorDist(wpc.pitchcontext[ix,:40], wpc.pitchcontext[ix,40:])
    return res

def computeNovelty(
    song: Song,
    wpc : PitchContext,
):
    """Computes for each note the 'novelty' of the following context with respect to the preceding context.
    Novelty for a pitch is computed as the percentual contribution of the following pitch value to the total
    of the preceding and following values.
    The overall novelty value is the average of novelty values of all pitches.

    Parameters
    ----------
    song : Song
        Ojbect with song data.
    wpc : PitchContext
        Object with pitch context data

    Returns
    -------
    numpy array
        1D numpy array with a novelty value for each note.
    """
    novelty = np.zeros( len(wpc.pitchcontext) )
    for ix in range(len(wpc.pitchcontext)):
        total = wpc.pitchcontext[ix,:40] + wpc.pitchcontext[ix,40:]
        new = wpc.pitchcontext[ix,40:]
        total[new==0] = 1
        perc  = new / total
        novelty[ix] = np.average(perc[perc>0])
    return novelty

def computeUnharmonicity(
    song: Song,
    wpc : PitchContext,
    dissonance: np.array,
    consonance: np.array,
    beatstrength_treshold: float,
    lastnoteharmonic: bool = False,
    epsilon: float = 10e-4
):
    """Computes for each note the degree to which it is 'unharmonic'.
    The unharmonicity value is the dissonance minus the consonance of a note in its context, with lower boundary of zero.
    Each note with beatstrength lower than beatstrength_threshold, and dissonant in its context is considered 'unharmonic' (value: 0.0).
    If parameter lastnoteharmonic is True, the last note gets unharmonicity value 0.0

    Parameters
    ----------
    song : Song
        Ojbect with song data.
    wpc : PitchContext
        Object with pitch context data
    dissonance : 1D numpy array
        Dissonance values as returned by computeDissonance
    consonance : 1D numpy array
        Consonance values as returned by computeConsonance
    beatstrength_threshold : float
        Consider all notes with beatstrength >= beatstrength_threshold as 'harmonic' (value 0.0).
    lastnoteharmonic : bool
        If True, always consider the last note of the melody as 'harmonic' (value 0.0).
        
    Returns
    -------
    numpy array
        1D numpy array with a unharmonicity value for each note.
    """
    beatstrength = song.mtcsong['features']['beatstrength']
    unharmonicity = np.zeros( len(wpc.pitchcontext) )
    for ix in range(len(wpc.pitchcontext)):
        if beatstrength[wpc.ixs[ix]] < beatstrength_treshold - epsilon:
            #unharmonicity[ix] = max ( dissonance[ix] - consonance[ix], 0.0 )
            if consonance[ix] == 0.0:
                unharmonicity[ix] = 10.0
            else:
                unharmonicity[ix] = max ( dissonance[ix] - consonance[ix], 0.0)
                #unharmonicity[ix] = dissonance[ix] / consonance[ix]
    #do last note separately
    if lastnoteharmonic:
        unharmonicity[-1] = 0.0  # e.g. NLB123866_01
    return unharmonicity

def computeAccented(song: Song, level, epsilon: float = 10e-4):
    return [True if song.mtcsong['features']['beatstrength'][ix] >= (level-epsilon) else False for ix in range(song.getSongLength())]


class ImpliedHarmony:

    def __init__(self, wpc: 'PitchContext'):
        self.wpc = wpc
        self.song = wpc.song
        self.songlength = self.song.getSongLength()
        self.params = wpc.params  
        self.numchords = 0 # will be set in self.initchords()
        self.chordquality = {} # will be set in self.initchords()
        self.initchords()

    def initchords(self):
        chordmask_dim = np.zeros(40, dtype=int)
        chordmask_min = np.zeros(40, dtype=int)
        chordmask_maj = np.zeros(40, dtype=int)
        chordmask_dom = np.zeros(40, dtype=int)

        chordmask_dim[[0, 11, 22]] = [1, 3, 5] #[root, third, fifth]
        chordmask_min[[0, 11, 23]] = [1, 3, 5] #[root, third, fifth]
        chordmask_maj[[0, 12, 23]] = [1, 3, 5] #[root, third, fifth]
        chordmask_dom[[0, 12, 23, 34]] = [1, 3, 5, 7] #[root, third, fifth, seventh]

        self.chordquality = {
            0: 'dim',
            1: 'm',
            2: '',
            3: '7'
        }
        self.masks = np.stack([chordmask_dim, chordmask_min, chordmask_maj, chordmask_dom])
        self.numchords = self.masks.shape[0]
        self.chordtones = np.zeros((40,40,self.numchords), dtype=int) #(pitch, root of chord, chord quality). 1: root, 3: third, 5: fifth, 7: seventh
        for chordq in range(self.numchords):
            for rootpitch in range(40):
                chordmask_shift = np.roll(self.masks, rootpitch, axis=1)
                self.chordtones[np.where(chordmask_shift[chordq]),rootpitch,chordq] = chordmask_shift[chordq][np.where(chordmask_shift[chordq])]

    def chordTransitionScore(
            self,
            chords,
            traceback,
            chord1_ixs,
            chord2_ixs,
            scalemask=np.ones(40, dtype=bool),
            song=None,
            wpc=None,
            ih=None
        ):
        # Basic implementation
        # Please, provide your own.

        #scoring scheme.
        pitch1 = chord1_ixs[1] % 40
        pitch2 = chord2_ixs[1] % 40
        #no score if root of chord tones is not in the scalemask)
        if not scalemask[pitch1] or not scalemask[pitch2]:
            return 0.0

        # start with score for 'next' chord
        score = chords[chord2_ixs]
        
        return score

    #ix = ix in full song
    def getP40(self, ix):
        return (self.song.mtcsong['features']['pitch40'][ix] - 1) % 40

    # if extendToAllNaturalTones, also include tones that are 'missing'. E.g. NLB070513_01 in G, but no F#
    # include all alterations. So, no F in melody, include Fb, F, and F#
    def getScaleMask(self, extendToAllNaturalTones=False):

        naturals = np.ones(7, dtype=bool) #[F C G D A E B] # True if stemtone not present in melody
        naturalixs = np.array([19, 2, 25, 8, 31, 14, 37])
        naturalsix = np.zeros(40, dtype=int)
        naturalsix[30:33] = 4 #A
        naturalsix[36:39] = 6 #B
        naturalsix[1:4]   = 1 #C
        naturalsix[7:10]  = 3 #D
        naturalsix[13:16] = 5 #E
        naturalsix[18:21] = 0 #F
        naturalsix[24:27] = 2 #G

        seq_length = len(self.wpc.ixs)

        #make scalemask for each note. includes the alteration of the stemtone that is CLOSEST to the focus note
        #TODO: different weight for looking back and looking forward?

        #first record at which index the closest same stemtone is in the melody
        mostrecent = np.zeros((seq_length, 7), dtype=int) + 1000 #gives the index of the most recent stemtone (1000 if not yet)
        mostnext = np.zeros((seq_length, 7), dtype=int) - 1000 #gives the index of the closes next occurrence of the scale tone (-1000 if not any more)
        #first pitch for mostrecent
        p40 = self.getP40(self.wpc.ixs[0])
        mostrecent[0][naturalsix[p40]] = 0
        #rest
        for ix in range(1, seq_length):
            mostrecent[ix] = mostrecent[ix-1]
            p40 = self.getP40(self.wpc.ixs[ix])
            mostrecent[ix][naturalsix[p40]] = ix
        #last pitch for mostnext
        p40 = p40 = self.getP40(self.wpc.ixs[-1])
        mostnext[-1][naturalsix[p40]] = seq_length - 1
        #rest
        for ix in reversed(range(seq_length-1)):
            mostnext[ix] = mostnext[ix+1]
            p40 = p40 = self.getP40(self.wpc.ixs[ix])
            mostnext[ix][naturalsix[p40]] = ix

        #now create the scales for each of the notes
        scalemask = np.zeros((seq_length, 40))

        for ix in range(seq_length):
            diff_prev = np.abs(ix - mostrecent[ix])
            diff_next = np.abs(mostnext[ix] - ix)
            next_ixs = np.where(diff_prev > diff_next)[0]
            prev_ixs = np.where(diff_prev <= diff_next)[0]
            #from prev context
            for scale_ix in prev_ixs:
                if mostrecent[ix][scale_ix] != 1000:
                    p40 = self.getP40( self.wpc.ixs[ mostrecent[ix][scale_ix] ] )
                    scalemask[ix, p40] = True
                    naturals[naturalsix[p40]] = False
            #take from next context
            for scale_ix in next_ixs:
                if mostnext[ix][scale_ix] != -1000:
                    p40 = self.getP40( self.wpc.ixs[ mostnext[ix][scale_ix] ] )
                    scalemask[ix, p40] = True
                    naturals[naturalsix[p40]] = False

        # scalemask = np.zeros(40)
        # for ix in range(self.wpc.pitchcontext.shape[0]):
        #     p40 = (self.song.mtcsong['features']['pitch40'][ix] - 1) % 40
        #     scalemask[p40] = True
        #     naturals[naturalsix[p40]] = False

        # now extend each 
        # missing stem tones should be missing in all scalemasks, but addition might differ

        if extendToAllNaturalTones:

            for seq_ix in range(seq_length):

                #add missing tones
                #flats
                for ix in sorted(list(np.where(naturals)[0])): #from F->B order is important, more than one tone might be missing. first fix tones far away from C
                    #figure out whether to add flat

                    missing       = naturalixs[ix]
                    missingflat   = missing - 1 
                    fifthdown     = naturalixs[(ix - 1) % 7]
                    fifthdownflat = fifthdown - 1
                    fifthup       = naturalixs[(ix + 1) % 7]
                    fifthupflat   = fifthup - 1
                    fifthupsharp  = fifthup + 1

                    #### SOLVE MINOR, leading tone (if not in melody)

                    if ix == 0:
                        #When Fb? Only if Cb is present
                        if scalemask[seq_ix,fifthupflat]:
                            scalemask[seq_ix,missingflat] = True
                        #when F? If no C#
                        if not scalemask[seq_ix,fifthupsharp]:
                            scalemask[seq_ix,missing] = True
                        continue
                    
                    if ix == 6: #Bb ?
                        #When Bb? If Eb is present or if F is present
                        if scalemask[seq_ix,fifthdownflat] or scalemask[seq_ix,fifthup]:
                            scalemask[seq_ix,missingflat] = True
                        #when B? always - We deal with B# below
                        scalemask[seq_ix,missing] = True
                        continue

                    if scalemask[seq_ix,fifthdownflat]:
                        scalemask[seq_ix,missingflat] = True
                    elif scalemask[seq_ix,fifthdown] and scalemask[seq_ix,fifthup]:
                        scalemask[seq_ix,missing] = True
                    elif scalemask[seq_ix,fifthdown] and scalemask[seq_ix,fifthupflat]:
                        scalemask[seq_ix,missingflat] = True
                        scalemask[seq_ix,missing] = True

                #sharps
                for ix in sorted(list(np.where(naturals)[0]), reverse=True): #from B->F order is important, more than one tone might be missing. first fix tones far away from C
                    
                    #figure out whether to add flat

                    missing        = naturalixs[ix]
                    missingsharp   = missing + 1 
                    fifthdown      = naturalixs[(ix - 1) % 7]
                    fifthdownsharp = fifthdown + 1
                    fifthup        = naturalixs[(ix + 1) % 7]
                    fifthupsharp   = fifthup + 1
                    fifthupflat    = fifthup - 1

                    if ix == 6: # B#?
                        #when B#? If E# present
                        if scalemask[seq_ix,fifthdownsharp]:
                            scalemask[seq_ix,missingsharp] = True
                        #when B? when No flats (i.e. no Eb)
                        if not scalemask[seq_ix,fifthdownflat]:
                            scalemask[seq_ix,missing] = True
                        continue

                    if ix == 0: # F#?
                        #when F#? If C# present or if B present
                        if scalemask[seq_ix,fifthupsharp] or scalemask[seq_ix,fifthdown]:
                            scalemask[seq_ix,missingsharp] = True
                        #when F? If Bb not present (deal with Bb above)
                        if not scalemask[seq_ix,fifthdownflat]:
                            scalemask[seq_ix,missing] = True
                        continue

                    if scalemask[seq_ix,fifthupsharp]:
                        scalemask[seq_ix,missingsharp] = True
                    elif scalemask[seq_ix,fifthup] and scalemask[seq_ix,fifthdown]:
                        scalemask[seq_ix,missing] = True
                    elif scalemask[seq_ix,fifthdownsharp] and scalemask[seq_ix,fifthup]:
                        scalemask[seq_ix,missingsharp] = True
                        scalemask[seq_ix,missing] = True

        return scalemask

    def getOptimalChordSequence(self, use_scalemask=True, chordTransitionScoreFunction=None):

        if chordTransitionScoreFunction == None:
            chordTransitionScoreFunction = self.chordTransitionScore

        if use_scalemask:
            scalemask = self.getScaleMask(extendToAllNaturalTones=True)
        else:
            scalemask = np.ones((len(self.wpc.ixs), 40), dtype=bool)
        
        chords      = self.getChords(use_scalemask=use_scalemask)
        numpitches  = chords.shape[1]        
        score       = np.zeros( (self.wpc.pitchcontext.shape[0], numpitches, self.numchords) )
        traceback   = np.zeros( (self.wpc.pitchcontext.shape[0], numpitches, self.numchords, 4), dtype=int ) #coordinates (pitch, chord, ix last root change, previous pitch) for previous chord
        trace       = np.zeros( (self.wpc.pitchcontext.shape[0], 2), dtype=int ) # (pitch, chord) for each note
        trace_score = np.zeros( (self.wpc.pitchcontext.shape[0], 2) ) # (score, score_diff) for each note

        #initialization: first note gets its own chords
        score[0] = chords[0]
        for ix in range(1, self.wpc.pitchcontext.shape[0]):
            #find indices of chord1
            chord1_ixs = np.where(chords[ix-1])
            #find indices of chord2
            chord2_ixs = np.where(chords[ix])

            #E.g., NLB191190_01, note 40?
            #if all available chords are forbidden (score -10.), another chord gets chosen (score 0)
            #TODO: all chord1_ixs should also be present in chord2_ixs, to always allow continuation of a chord?
            #TODO: better: only take maximum over scores in chord2_ixs

            for ixs2 in zip(chord2_ixs[0], chord2_ixs[1]):
                allscores = np.zeros( (numpitches, self.numchords) )
                for ixs1 in zip(chord1_ixs[0], chord1_ixs[1]):
                    transistionscore = chordTransitionScoreFunction(
                        chords,
                        traceback,
                        (ix-1,ixs1[0],ixs1[1]),
                        (ix, ixs2[0], ixs2[1]),
                        scalemask=scalemask[ix],
                        song=self.song,
                        wpc=self.wpc,
                        ih=self,
                    )
                    allscores[ixs1] = score[ix-1, ixs1[0], ixs1[1]] + transistionscore
                #now find max
                max_ixs = np.unravel_index(np.argmax(allscores), allscores.shape)
                maxscore = allscores[max_ixs]
                #update score
                score[ix, ixs2[0], ixs2[1]] = maxscore
                #update traceback
                traceback[ix, ixs2[0], ixs2[1]][0:2] = max_ixs
                if ixs2[0] % 40 != max_ixs[0] % 40: #new root @ ix
                    traceback[ix, ixs2[0], ixs2[1]][2] = ix
                    traceback[ix, ixs2[0], ixs2[1]][3] = max_ixs[0]
                else:
                    traceback[ix, ixs2[0], ixs2[1]][2] = traceback[ix-1, max_ixs[0], max_ixs[1]][2]  #take start of root from prvious in trace
                    traceback[ix, ixs2[0], ixs2[1]][3] = traceback[ix-1, max_ixs[0], max_ixs[1]][3]  #take previous pitch from prvious in trace


        #now do the traceback.
        #find max score for last note
        max_ixs = np.unravel_index(np.argmax(score[-1]), score[-1].shape)
        trace[-1] = (max_ixs[0], max_ixs[1])
        trace_score[-1] = (score[-1,max_ixs[0], max_ixs[1]], 0)
        for ix in range(self.wpc.pitchcontext.shape[0]-2, -1, -1):
            trace_ixs = traceback[ix+1,trace[ix+1][0],trace[ix+1][1]]
            thisscore = score[ix,trace_ixs[0], trace_ixs[1]]
            trace[ix] = (trace_ixs[0], trace_ixs[1])
            trace_score[ix][0] = thisscore

        for ix in range(1, self.wpc.pitchcontext.shape[0]):
            trace_score[ix][1] = trace_score[ix][0] - trace_score[ix-1][0]

        return trace, trace_score, score, traceback

    def trace2str(self, trace, contextType=False):
        if contextType:
            return [base40[trace[ix][0]%40] + self.chordquality[trace[ix][1]] + str(int(trace[ix][0]/40)) for ix in range(self.wpc.pitchcontext.shape[0])]
        else:
            return [base40[trace[ix][0]%40] + self.chordquality[trace[ix][1]] for ix in range(self.wpc.pitchcontext.shape[0])]

    def getChords(self, use_scalemask=True):
        seq_length = len(self.wpc.ixs)
        #prepare data structure:
        if use_scalemask:
            scalemask = self.getScaleMask(extendToAllNaturalTones=True)
        else:
            scalemask = np.ones((seq_length, 40), dtype=bool)
        numpitches = 120 # 40 pre, 40 post, 40 all
        chords = np.zeros( (self.wpc.pitchcontext.shape[0], numpitches, self.numchords ) ) # number of notes, 40 pre-pitches+40post-pitches, 4 chords (dim,min,maj,dom)
        for ix in range(self.wpc.pitchcontext.shape[0]):
            scores, strengths = self.getChordsForNote(self.wpc.pitchcontext[ix], normalize=True, scalemask=scalemask[ix])
            chords[ix] = np.multiply(
                scores,
                strengths
            )
        return chords

    def getChordsForNote(self, pitchcontextvector, normalize=True, scalemask=np.ones(40, dtype=bool)):
        #find out whether pitches could be arranged as series of thirds
        
        epsilon = 10e-4
        chordmask_minseventh = np.zeros(40)
        np.put(chordmask_minseventh, [34], 1.0) #used for check presence seventh in dom chord

        #only take natural tones, one b or one # as root
        valid_shifts = [1, 2, 3, 7, 8, 9, 13, 14, 15, 18, 19, 20, 24, 25, 26, 30, 31, 32, 36, 37, 38]

        #self.masks contains integers indicating the function of the tone in the chord (root, third, fifht, seventh)
        #here we need masks with ones everywhere
        binarymasks = np.clip(self.masks, 0, 1)

        #get a value for every rotation of the chordmasks
        score_pre = np.zeros((40, self.numchords))
        score_post = np.zeros((40, self.numchords))
        score_all = np.zeros((40, self.numchords))
        strength_pre = np.zeros((40, self.numchords))
        strength_post = np.zeros((40, self.numchords))
        strength_all = np.zeros((40, self.numchords))
        for shift in range(40):
            if not shift in valid_shifts:
                continue

            chordmask_shift = np.roll(binarymasks, shift, axis=1)
            chordmask_minseventh_shift = np.roll(chordmask_minseventh, shift)

            #only accept chords which have all notes in the (local) scale
            for maskid in range(self.numchords):
                if np.prod(scalemask[np.where(chordmask_shift[maskid])]) < epsilon:
                    chordmask_shift[maskid] = 0

            score_pre[shift] = np.sum(np.multiply(pitchcontextvector[:40],chordmask_shift), axis=1)
            if np.sum(pitchcontextvector[:40]) > epsilon:
                strength_pre[shift] = score_pre[shift] / np.sum(pitchcontextvector[:40])
            score_post[shift] = np.sum(np.multiply(pitchcontextvector[40:],chordmask_shift), axis=1)
            if np.sum(pitchcontextvector[40:]) > epsilon:
                strength_post[shift] = score_post[shift] / np.sum(pitchcontextvector[40:])
            score_all[shift] = np.sum(np.multiply(pitchcontextvector[:40]+pitchcontextvector[40:],chordmask_shift), axis=1)
            if np.sum(pitchcontextvector) > epsilon:
                strength_all[shift] = score_all[shift] / np.sum(pitchcontextvector)

            #if seventh in dom chord is not present -> erase dom chord
            if np.sum(np.multiply(chordmask_minseventh_shift,pitchcontextvector[:40])) < epsilon:
                score_pre[shift][3] = 0.0
                strength_pre[shift][3] = 0.0
            if np.sum(np.multiply(chordmask_minseventh_shift,pitchcontextvector[40:])) < epsilon:
                score_post[shift][3] = 0.0
                strength_post[shift][3] = 0.0
            if np.sum(np.multiply(chordmask_minseventh_shift,pitchcontextvector[:40]+pitchcontextvector[:40])) < epsilon:
                score_all[shift][3] = 0.0
                strength_all[shift][3] = 0.0

        if normalize:
            if np.sum(score_pre) > 0.0:
                score_pre  = score_pre / np.max(score_pre)
            if np.sum(score_post) > 0.0:
                score_post = score_post / np.max(score_post)
            if np.sum(score_all) > 0.0:
                score_all  = score_all / np.max(score_all)

        scores    = np.concatenate( (score_pre, score_post, score_all) )
        strengths = np.concatenate( (strength_pre, strength_post, strength_all) )

        return scores, strengths
    
    def printChordsForNote(self, chords):
        epsilon = 10e-4
        chordixs = np.where(chords > epsilon)
        chords_pre = []
        chords_post = []
        chords_all = []

        for ix in zip(chordixs[0], chordixs[1]):
            info = (base40[ix[0] % 40], self.chordquality[ix[1]], chords[ix])
            if ix[0] < 40:
                chords_pre.append(info)
            else:
                if ix[0] < 80:
                    chords_post.append(info)
                else:
                    chords_all.append(info)
        
        print('pre:')
        for info in sorted(chords_pre, key=lambda x: x[2], reverse=True):
            print(info)

        print('post:')
        for info in sorted(chords_pre, key=lambda x: x[2], reverse=True):
            print(info)

        print('all:')
        for info in sorted(chords_pre, key=lambda x: x[2], reverse=True):
            print(info)

    def printChord(self, chord_in):
        chordixs = np.where(chord_in > 0)
        chords = []
        
        for ix in zip(chordixs[0], chordixs[1]):
            pitch = ix[0] % 40
            chords.append((ix[0], base40[pitch], self.chordquality[ix[1]], chord_in[ (ix[0], ix[1]) ]))
        for chord in sorted(chords, key=lambda x: x[3], reverse=True):
            print(chord)
    
    #chords: output of getChords
    #transition from ix1 to ix2
    def printAllTransitions(self, chords, ix1, ix2, scores=None, chordTransitionScoreFunction=None, use_scalemask=True):
        if chordTransitionScoreFunction == None:
            chordTransitionScoreFunction = self.chordTransitionScore
        if scores is None:
            scores = np.zeros([chords.shape[0], chords.shape[1], self.numchords])
        
        #find indices of chord1
        chord1_ixs = np.where(chords[ix1])
        #find indices of chord2
        chord2_ixs = np.where(chords[ix2])

        if use_scalemask:
            scalemask = self.getScaleMask(extendToAllNaturalTones=True)
        else:
            scalemask = np.ones((len(self.wpc.ixs), 40), dtype=bool)

        transitions = []
        for ixs2 in zip(chord2_ixs[0], chord2_ixs[1]):
            for ixs1 in zip(chord1_ixs[0], chord1_ixs[1]):
                transistionscore = chordTransitionScoreFunction(
                    chords,
                    (ix1,ixs1[0],ixs1[1]),
                    (ix2, ixs2[0], ixs2[1]),
                    scalemask=scalemask[ix2],
                    song=self.song,
                    wpc=self.wpc
                )
                side1 = 'pre'
                if ixs1[0] > 39:
                    if ixs1[0] > 79:
                        side1 = 'all'
                    else:
                        side2 = 'post'
                side2 = 'pre'
                if ixs2[0] > 39:
                    if ixs2[0] > 79:
                        side2 = 'all'
                    else:
                        side2 = 'post'
                #report transition:
                transitions.append((
                    base40[ixs1[0] % 40],
                    self.chordquality[ixs1[1]],
                    side1,
                    '->',
                    base40[ixs2[0] % 40],
                    self.chordquality[ixs2[1]],
                    side2,
                    transistionscore,
                    scores[ix1, ixs1[0], ixs1[1]],
                    transistionscore + scores[ix1, ixs1[0], ixs1[1]],
                ))
        for tr in sorted(transitions, key=lambda x: x[9], reverse=True):
            print(' '.join([str(t) for t in tr]))

    def printTrace(self, trace, traceback):
        for ix, tr in enumerate(trace):
            print(
                ix,
                base40[trace[ix][0]%40],
                self.chordquality[trace[ix][1]],
                traceback[
                    ix,
                    trace[ix][0],
                    trace[ix][1]
                ]
            )

