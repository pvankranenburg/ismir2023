"""Provides a user interface for exploring implied chords model

Run as:

$ streamlit run st_impliedchords.py -- -krnpath <path_to_kern_files> -jsonpath <path_to_json_files>
"""

import argparse
import json
from fractions import Fraction
from PIL import Image
import os
import ast

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from IPython import display
import numpy as np
from pitchcontext import Song, PitchContext
from pitchcontext.models import ImpliedHarmony

epsilon = 10e-4

def params2dict(textfield):
    if len(textfield.strip()) > 0:
        lines = textfield.split('\n')
        return { pair[0] : ast.literal_eval(pair[1]) for pair in [line.split('=') for line in lines if '=' in line] }
    else:
        return dict()

parser = argparse.ArgumentParser(description='Generate a chord sequence for a given melody.')
parser.add_argument(
    '-krnpath',
    dest='krnpath',
    help='Path to **kern files.',
    default="testcorpus/krn",
)
parser.add_argument(
    '-jsonpath',
    dest='jsonpath',
    help='Path to json files (in MTCFeatures format).',
    default="testcorpus/json",
)
args = parser.parse_args()
krnpath = args.krnpath
jsonpath = args.jsonpath

st.set_page_config(layout="wide")
col1, col2 = st.columns([2,1])

#select first file from the krnpath
krnfiles = os.listdir(krnpath)
for krnfile in krnfiles:
    if krnfile.endswith('.krn'):
        break
else:
    krnfile = 'none'

firstid = krnfile.rstrip(".krn")


#(widgetname, variablename, default value)
widgets_defaults = [
    ('songid_wid',                      'songid',                           firstid),
    ('same_root_wid',                   'same_root_slider',                 0.1),
    ('diff_root_wid',                   'diff_root_slider',                 0.8),
    ('granularity_threshold_wid',       'granularity_threshold',            0.5),
    ('focuschordtone_threshold_wid',    'focuschordtone_threshold',         0.5),
    ('allow_appoggiatura_wid',          'allow_appoggiatura_check',         True),
    ('syncope_level_threshold_wid',     'syncope_level_threshold',          1.0),
    ('upbeat_change_wid',               'upbeat_change_check',              True),
    ('allowmajdom_wid',                 'allowmajdom_check',                True),
    ('use_scalemask_wid',               'use_scalemask_check',              True),
    ('no_fourth_fifth_wid',             'no_fourth_fifth_slider',           0.75),
    ('root_third_final_wid',            'root_third_final_check',           True),
    ('final_third_wid',                 'final_third_slider',               0.75),
    ('final_v_i_wid',                   'final_v_i_slider',                 0.1),
    ('final_iv_i_wid',                  'final_iv_i_slider',                0.8),
    ('dom_fourth_wid',                  'dom_fourth_slider',                0.1),
    ('dim_m2_wid',                      'dim_m2_slider',                    0.1),
    ('fourth_dom_wid',                  'fourth_dom_slider',                0.8),
    ('pre_c_wid',                       'pre_c_slider',                     3.0),
    ('post_c_wid',                      'post_c_slider',                    3.0),
    ('preauto_wid',                     'preauto_check',                    True),
    ('postauto_wid',                    'postauto_check',                   True),
    ('context_boundary_threshold_wid',  'context_boundary_threshold_radio', 1.0),
    ('context_rel_focus_wid',           'context_rel_focus_check',          False),
    ('partialnotes_wid',                'partialnotes_check',               True),
    ('accweight_wid',                   'accweight_check',                  True),
    ('include_focus_pre_wid',           'include_focus_pre_check',          True),
    ('include_focus_post_wid',          'include_focus_post_check',         True),
    ('pre_usemw_wid',                   'pre_usemw_check',                  True),
    ('post_usemw_wid',                  'post_usemw_check',                 True),
    ('pre_usedw_wid',                   'pre_usedw_check',                  True),
    ('post_usedw_wid',                  'post_usedw_check',                 True),
    ('mindistw_pre_wid',                'mindistw_pre_slider',              0.0),
    ('mindistw_post_wid',               'mindistw_post_slider',             0.0),
]

def delSessionState(delsongid=False):
    for wid in widgets_defaults:
        if not delsongid:
            if wid[0] == 'songid_wid':
                continue
        del st.session_state[wid[0]]
    delParams()

def delParams():
    st.session_state.params_wid = ''

def newSong():
    delParams()

with st.sidebar:

    st.button(
        label='Restore defaults',
        on_click=delSessionState
    )

    params_area = st.text_area(
        'Parameter setting',
        key='params_wid'
    )

    paramdict = params2dict(params_area)

    # Check for provided param setting
    if 'krnpath' in paramdict.keys():
        krnpath = paramdict['krnpath']
    if 'jsonpath' in paramdict.keys():
        jsonpath = paramdict['jsonpath']
    for wid in widgets_defaults:
        if wid[1] in paramdict.keys():
            st.session_state[wid[0]] = paramdict[wid[1]]
        else:
            if wid[0] not in st.session_state:
                st.session_state[wid[0]] = wid[2]        

    songid = st.text_input(
        label="Song ID",
        key='songid_wid',
        on_change=newSong,
    )

    #we need to load the song here, because the song is needed to set pre_c_slider and post_c_slider max
    #but not necessary if params_area is provided
    krnfilename = os.path.join(krnpath, songid+'.krn')
    jsonfilename = os.path.join(jsonpath, songid+'.json')
    with open(jsonfilename,'r') as f:
        mtcsong = json.load(f)

    song = Song(mtcsong, krnfilename)
    songlength_beat = float(sum([Fraction(length) for length in song.mtcsong['features']['beatfraction']]))

    granularity_threshold = st.radio(    
        "Don't allow change on notes with beatstrength < (0: allow all)",
        (1.0, 0.5, 0.25, 0.125, 0),
        key='granularity_threshold_wid',
        on_change=delParams
    )

    syncope_level_threshold = st.radio(
        "Allow chord syncopes from position with beatstrength >=",
        (1.0, 0.5, 0.25, 0.125),
        key='syncope_level_threshold_wid',
        on_change=delParams,
    )

    upbeat_change_check = st.checkbox(
        "Above, except for an upbeat.",
        key='upbeat_change_wid',
        on_change=delParams,
    )

    focuschordtone_threshold = st.radio(    
        "Melody note must be chord tone at chord change if beatstrength >= (0: no restriction)",
        (1.0, 0.5, 0.25, 0.125, 0),
        key='focuschordtone_threshold_wid',
        on_change=delParams
    )

    allow_appoggiatura_check = st.checkbox(
        "Above, but allow appoggiatura.",
        key='allow_appoggiatura_wid',
        on_change=delParams,
    )

    use_scalemask_check = st.checkbox(
        "Use scale when choosing chords",
        key='use_scalemask_wid',
        on_change=delParams,
    )

    diff_root_slider = st.slider(
        'Multiplier root change',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key='diff_root_wid',
        on_change=delParams
    )

    same_root_slider = st.slider(
        'Multiplier same root but different quality',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key='same_root_wid',
        on_change=delParams
    )

    allowmajdom_check = st.checkbox(
        "above, but except maj->dom with the same root.",
        key='allowmajdom_wid',
        on_change=delParams,
    )

    no_fourth_fifth_slider = st.slider(
        'Multiplier root movement other than 4th or 5th',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key='no_fourth_fifth_wid',
        on_change=delParams,
    )

    dom_fourth_slider = st.slider(
        'Multiplier NO 4th up after dominant',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key='dom_fourth_wid',
        on_change=delParams,
    )

    dim_m2_slider = st.slider(
        'Multiplier NO minor second after dim',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key='dim_m2_wid',
        on_change=delParams,
    )

    fourth_dom_slider = st.slider(
        'Multiplier NO major or dominant before 4th up',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key='fourth_dom_wid',
        on_change=delParams,
    )

    root_third_final_check = st.checkbox(
        "Chord-root or third in melody at final note",
        key='root_third_final_wid',
        on_change=delParams,
    )

    final_third_slider = st.slider(
        'Multiplier third at final note.',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key='final_third_wid',
        on_change=delParams,
    )

    final_v_i_slider = st.slider(
        'Multiplier NO fifth or fourth up on final pitch',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key='final_v_i_wid',
        on_change=delParams,
    )
    final_iv_i_slider = st.slider(
        'Multiplier fifth up on final pitch',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key='final_iv_i_wid',
        on_change=delParams,
    )

    showcontexttype_check = st.checkbox(
        "Show context with chord (0\:pre, 1\:post, 2\:both)",
        key='showcontexttype_wid',
        value=False,
    )

    pre_c_slider = st.slider(
        'Length of preceding context (beats)',
        min_value=0.0,
        max_value=songlength_beat,
        step=0.5,
        key='pre_c_wid',
        on_change=delParams,
    )
    post_c_slider = st.slider(
        'Length of following context (beats)',
        min_value=0.0,
        max_value=songlength_beat,
        step=0.5,
        key='post_c_wid',
        on_change=delParams,
    )
    preauto_check = st.checkbox(
        "Determine preceding context automatically.",
        key='preauto_wid',
        on_change=delParams,
    )
    postauto_check = st.checkbox(
        "Determine following context automatically",
        key='postauto_wid',
        on_change=delParams,
    )
    context_boundary_threshold_radio = st.radio(    
        "Boundary for automatic context:",
        (1.0, 0.5),
        key='context_boundary_threshold_wid',
        on_change=delParams
    )
    context_rel_focus_check = st.checkbox(
        "Never extend context beyond note with higher weight than focus note",
        key='context_rel_focus_wid',
        on_change=delParams,
    )
    partialnotes_check = st.checkbox(
        "Include partial notes in preceding context.",
        key='partialnotes_wid',
        on_change=delParams,
    )
    accweight_check = st.checkbox(
        "Accumulate Weight.",
        key='accweight_wid',
        on_change=delParams,
    )
    include_focus_pre_check = st.checkbox(
        "Include Focus note in preceding context:",
        key='include_focus_pre_wid',
        on_change=delParams,
    )
    include_focus_post_check = st.checkbox(
        "Include Focus note in following context:",
        key='include_focus_post_wid',
        on_change=delParams,
    )
    pre_usemw_check = st.checkbox(
        "Use metric weight for preceding context",
        key='pre_usemw_wid',
        on_change=delParams,
    )
    post_usemw_check = st.checkbox(
        "Use metric weight for following context",
        key='post_usemw_wid',
        on_change=delParams,
    )
    pre_usedw_check = st.checkbox(
        "Use distance weight for preceding context",
        key='pre_usedw_wid',
        on_change=delParams,
    )
    post_usedw_check = st.checkbox(
        "Use distance weight for following context",
        key='post_usedw_wid',
        on_change=delParams,
    )
    mindistw_pre_slider = st.slider(
        'Minimal distance weight preceding context',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key='mindistw_pre_wid',
        on_change=delParams,
    )
    mindistw_post_slider = st.slider(
        'Minimal distance weight following context',
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key='mindistw_post_wid',
        on_change=delParams,
    )


len_context_pre = 'auto' if preauto_check else pre_c_slider
len_context_post = 'auto' if postauto_check else post_c_slider

syncopes=True
if accweight_check:
    syncopes=False

wpc = PitchContext(
    song,
    syncopes=syncopes,
    remove_repeats=False,
    accumulate_weight=accweight_check,
    partial_notes=partialnotes_check,
    context_type='beats',
    len_context_pre=len_context_pre,
    len_context_post=len_context_post,
    len_context_params={
        'threshold':context_boundary_threshold_radio,
        'not_heigher_than_focus': context_rel_focus_check
    },
    use_metric_weights_pre=pre_usemw_check,
    use_metric_weights_post=post_usemw_check,
    include_focus_pre=include_focus_pre_check,
    include_focus_post=include_focus_post_check,
    use_distance_weights_pre=pre_usedw_check,
    use_distance_weights_post=post_usedw_check,
    min_distance_weight_pre=mindistw_pre_slider,
    min_distance_weight_post=mindistw_post_slider,
)

# Chord Transition Score
def myChordTransitionScore(
        chords,
        traceback,
        chord1_ixs,
        chord2_ixs,
        scalemask=np.ones(40, dtype=bool),
        song=None,
        wpc=None,
        ih=None,
    ):

    #scoring scheme.
    root1 = chord1_ixs[1] % 40
    root2 = chord2_ixs[1] % 40
    songlength = song.songlength
    shift = (root2 - root1) % 40 #interval of roots in base40
    #ix last root change
    ix_lastchange = traceback[chord1_ixs[0],chord1_ixs[1],chord1_ixs[2]][2]
    pitch_lastchange = traceback[chord1_ixs[0],chord1_ixs[1],chord1_ixs[2]][3]
    shift_lastchange = (root2 - (pitch_lastchange % 40)) % 40
    #pitch of note1 and note2
    pitch1 = (song.mtcsong['features']['pitch40'][wpc.ixs[chord1_ixs[0]]] - 1) % 40
    pitch2 = (song.mtcsong['features']['pitch40'][wpc.ixs[chord2_ixs[0]]] - 1) % 40

    #no score if root of chord tones is not in the scalemask)
    if not scalemask[root1] or not scalemask[root2]:
        return 0.0

    #else compute score step by step
    
    # Start with score for 'next' chord
    score = chords[chord2_ixs]

    # No chord change if note 2 not part of chord 2. Only on downbeat. No seventh as melody note.
    # allowance for appoggiatura if NEXT note does fit in chord, has lower beatstrength, and is second lower
    appoggiatura = False
    if allow_appoggiatura_check:
        dp2 = song.mtcsong['features']['diatonicpitch'][wpc.ixs[chord2_ixs[0]]]
        if wpc.ixs[chord2_ixs[0]] < songlength-1:
            pitch3 = (song.mtcsong['features']['pitch40'][wpc.ixs[chord2_ixs[0]]+1] - 1) % 40
            bs3 = song.mtcsong['features']['beatstrength'][wpc.ixs[chord2_ixs[0]]+1]
            dp3 = song.mtcsong['features']['diatonicpitch'][wpc.ixs[chord2_ixs[0]]+1]
        else:
            pitch3 = 0
            bs3 = 0
            dp3 = 0
        if ih.chordtones[pitch3, root2, chord2_ixs[2]] and bs3 < song.mtcsong['features']['beatstrength'][chord2_ixs[0]] and dp3-dp2 == -1:
            appoggiatura = True
    
    if focuschordtone_threshold > 0:
        if song.mtcsong['features']['beatstrength'][chord2_ixs[0]] >= focuschordtone_threshold-epsilon:
            if root2 != root1: # chord change
                if allow_appoggiatura_check and appoggiatura:
                    pass
                elif not ih.chordtones[pitch2, root2, chord2_ixs[2]] in [1, 3, 5]: #do not allow seventh as melody note
                    return -10.0

    # prefer continuation of the chord
    if root1 != root2:
        score = score * diff_root_slider

    # Discourage same root, different quality #except maj -> dom
    if root1 == root2:
        if chord1_ixs[2] != chord2_ixs[2]:
            if  not ( chord1_ixs[2] == 2 and chord2_ixs[2] == 3 ):
                score = score * same_root_slider

    # Prevent chord syncope (start at low metric weight (<secondary accent), continue past higher metric weight)
    # Relate this to last root CHANGE (is in traceback matrix)
    # not for upbeat (if upbeat_change_check)
    if not ( upbeat_change_check and float(Fraction(song.mtcsong['features']['beatinsong'][ix_lastchange])) < 0 ):
        if song.mtcsong['features']['beatstrength'][ix_lastchange] < ( syncope_level_threshold - epsilon ):
            if song.mtcsong['features']['beatstrength'][chord2_ixs[0]] > song.mtcsong['features']['beatstrength'][ix_lastchange]:
                #stimulate chord change at higher metric weight
                if root1 == root2:
                    return -10.

    # discourage root, and/or quality change on note with low metric weight (except maj->dom on same root)
    if granularity_threshold > 0:
        if song.mtcsong['features']['beatstrength'][chord2_ixs[0]] < granularity_threshold-epsilon:
            if allowmajdom_check:
                if (root1 != root2) or ( (root1 == root2) and ((chord1_ixs[2] != chord2_ixs[2]) and not (chord1_ixs[2] == 2 and chord2_ixs[2] == 3)) ):
                    return -10.
            else:
                if root1 != root2:
                    return -10.

    # prefer root movement of fourth and fifth (or continuation)
    if shift != 17 and shift != 23 and shift !=0:
        score = score * no_fourth_fifth_slider

    # prefer fifth up (perfect cadence) or fourth up (plagal cadence) as final root movement.
    # Consider last chord change.
    if chord2_ixs[0] == songlength - 1: #last note
        if root1 == root2:
            if shift_lastchange != 17 and shift_lastchange !=23:
                score = score * final_v_i_slider
            if shift_lastchange == 23:
                score = score * final_iv_i_slider
        else:
            if shift != 17 and shift != 23:
                score = score * final_v_i_slider
            if shift == 23:
                score = score * final_iv_i_slider

    # If previous is dom. Then root must be fourth up
    # Except if next is continuation of the dominant/major
    if chord1_ixs[2] == 3:
        if shift != 17 and shift != 0:
            score = score * dom_fourth_slider

    # If previous is dim. Then root must be semitone up
    # Except if next is continuation of the dim
    if chord1_ixs[2] == 0:
        if shift != 5 and shift != 0:
            score = score * dim_m2_slider

    # if root is fourth up: prefer maj or dom for first chord
    if shift == 17:
        if chord1_ixs[2] == 0 or chord1_ixs[2] == 1:
            score = score * fourth_dom_slider

    # prefer root or third in melody for last note
    if root_third_final_check:
        if chord2_ixs[0] == songlength - 1:
            if ih.chordtones[pitch2, root2, chord2_ixs[2]] == 1:
                pass
            elif ih.chordtones[pitch2, root2, chord2_ixs[2]] == 3:
                score = score * final_third_slider #small penalty for third in melody
            else:
                return -10.

    return score

ih = ImpliedHarmony(wpc)

trace, trace_score, score, traceback = ih.getOptimalChordSequence(chordTransitionScoreFunction=myChordTransitionScore, use_scalemask=use_scalemask_check)

strtrace = ih.trace2str(trace, contextType=showcontexttype_check)

#replace same chord
toremove = []
for ix in range(1, len(strtrace)):
    if strtrace[ix] == strtrace[ix-1]:
        toremove.append(ix)
for ix in toremove:
    strtrace[ix] = ' '

#replace - with b
for ix in range(len(strtrace)):
    strtrace[ix] = strtrace[ix].replace('-','b')

#ih.printTrace(trace, traceback)

with col1:
    pngfn_chords = song.createPNG(
        '/tmp',
        showfilename=False,
        filebasename=songid+'_harm',
        lyrics=strtrace,
        lyrics_ixs=wpc.ixs
    )
    image = Image.open(pngfn_chords)
    st.image(image, output_format='PNG', use_column_width=True)

    st.text(" ") 
    st.text(" ") 

    pngfn_orig = song.createPNG(
        '/tmp',
        showfilename=False
    )
    image = Image.open(pngfn_orig)
    st.image(image, output_format='PNG', use_column_width=True)

    #write parameters
    st.subheader("Parameter settings")
    st.text(f"{songid=}")
    st.text(f"{krnpath=}")
    st.text(f"{jsonpath=}")
    st.text(f"{granularity_threshold=}")
    st.text(f"{syncope_level_threshold=}")
    st.text(f"{focuschordtone_threshold=}")
    st.text(f"{allow_appoggiatura_check=}")
    st.text(f"{upbeat_change_check=}")
    st.text(f"{diff_root_slider=}")
    st.text(f"{same_root_slider=}")
    st.text(f"{allowmajdom_check=}")
    st.text(f"{root_third_final_check=}")
    st.text(f"{final_third_slider=}")
    st.text(f"{use_scalemask_check=}")
    st.text(f"{no_fourth_fifth_slider=}")
    st.text(f"{final_v_i_slider=}")
    st.text(f"{final_iv_i_slider=}")
    st.text(f"{dom_fourth_slider=}")
    st.text(f"{dim_m2_slider=}")
    st.text(f"{fourth_dom_slider=}")
    st.text(f"{pre_c_slider=}")
    st.text(f"{post_c_slider=}")
    st.text(f"{preauto_check=}")
    st.text(f"{postauto_check=}")
    st.text(f"{context_boundary_threshold_radio=}")
    st.text(f"{partialnotes_check=}")
    st.text(f"{accweight_check=}")
    st.text(f"{include_focus_pre_check=}")
    st.text(f"{include_focus_post_check=}")
    st.text(f"{pre_usemw_check=}")
    st.text(f"{post_usemw_check=}")
    st.text(f"{pre_usedw_check=}")
    st.text(f"{post_usedw_check=}")
    st.text(f"{mindistw_pre_slider=}")
    st.text(f"{mindistw_post_slider=}")

with col2:
    #reconstruct initial score:
    chords = ih.getChords()
    initialscore = []
    beatstrengths = []
    for ix in wpc.ixs:
        initialscore.append(chords[ix,trace[ix][0],trace[ix][1]])
        beatstrengths.append(song.mtcsong['features']['beatstrength'][wpc.ixs[ix]])

    report = wpc.printReport(
        beatstrength = beatstrengths,
        initialscore = initialscore,
        chordscore = [tr[1] for tr in trace_score]
    )
    components.html(f"<pre>{report}</pre>", height=650, scrolling=True)

