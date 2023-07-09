"""Provides class Cong, which encapsulates details about the melodic data."""

import copy
from fractions import Fraction
from math import gcd
import subprocess
import tempfile
import os
import json

import numpy as np
import music21 as m21
m21.humdrum.spineParser.flavors['JRP'] = True #For triplets
from IPython import display

#from datetime import datetime
#print(__file__, datetime.now().strftime("%H:%M:%S"))

#Exception: parsing failed
class ParseError(Exception):
    def __init__(self, arg):
        self.args = arg
    def __str__(self):
        return repr(self.value)

def lcm(a, b):
    """Computes the lowest common multiple.
    
    Parameters
    ----------
    a : int
    b : int

    Returns
    ------
    int
        Lowest common multiple of a and b.
    """
    return a * b // gcd(a, b)

def fraction_gcd(x, y):
    """Computes the greatest common divisor as Fraction
    
    Parameters
    ----------
    x : Fraction
    y : Fraction

    Returns
    ------
    Fraction
        greatest common divisor of x and y as Fraction.
    """
    a = x.numerator
    b = x.denominator
    c = y.numerator
    d = y.denominator
    return Fraction(gcd(a, c), lcm(b, d))

class Song:
    """Class for containing data about a song. An object of the class `Song` holds
    feature values for each note as provided by MTCFeatures, additionally computed
    features for computing the weighted pitch vectors, a music21 object representing
    the score.
    
    Parameters
    ----------
    mtcsong : dict
        Dictionary with feature values of all notes of the song, as provided by
        MTCFeatures
    krnfilename : string
        Filename of a corresponding **kern file

    Attributes
    ----------
    reduced : bool
        True if this song object is derived from another song object by removing notes.
    mtcsong : dict
        Dictionary with feature values of all notes of the song, as provided by
        MTCFeatures. The values of 'onsettick' are replaced by newly computed onsets.
    krnfilename : string
        Filename of a corresponding **kern file
    s : music21 Stream
        music21 object representing the score of the song. The following operation have
        been performed: ties removal, padding of incomplete bars, grace note removal.
        The  indices of the notes in the resulting stream should corresponsd to the indices
        of the feature values in `self.mtcsong`
    onsets : list of ints
        list with onset times. onsets[n] is the onset of the (n+1)th note. The term onset
        refers to the time in the score at which the note starts (in music21 this is offset).
        The resolution is determined by the number of ticks per quarter note
        as computed by getResolution().
    beatstrength_grid : list
        Contains a beatstrength for each possible onset as computed by the music21 meter model.
        Onsets are indices in the list. beatstrength_grid[n] is the beatstrength for a note
        starting at n.
    """

    def __init__(self, mtcsong, krnfilename, s_in=None, beatstrength_grid_in=None):
        """Instantiate an object of class Song.

        Parameters
        ----------
        mtcsong : dict
            Dictionary of feature values and metadata of a song as provided by MTCFeatures
        krnfilename : str
            Full file name of a corresponding **kern file.
            If krnfilename==None: derive from existing Song object (s_in and beatstrength_grid_in need to be provided.)
        s_in : m21.Stream
            If provided, do not parse from .krn file, but use the provided stream instead.
        beatstrength_grid_in : list (float)
            If provided, do not compute from .krn file, but use provided grid instead.
        """
        self.mtcsong = copy.deepcopy(mtcsong)
        if krnfilename != None:
            self.reduced = False
            self.krnfilename = krnfilename
            self.s = self.parseMelody()
            self.onsets = self.getOnsets()
            self.beatstrength_grid = self.create_beatstrength_grid()
        else:
            self.reduced = True
            self.krnfilename = None
            self.s = copy.deepcopy(s_in)
            self.onsets = self.getOnsets()
            self.beatstrength_grid = copy.deepcopy(beatstrength_grid_in)
        self.songlength = None
        #always do this: recomputes features for reduced Song objects
        self.add_features()

    def getSongLength(self):
        """Returns the number of notes in the song

        Returns
        -------
        int
            Number of notes in the song
        """
        if self.songlength == None:
            self.computeSongLength()
        return self.songlength

    def computeSongLength(self):
        self.songlength = len(self.mtcsong['features']['pitch'])

    def getDurationUnit(self):
        """Returns a unit of note duration that is the greatest common divisor of all note durations.

        Returns
        -------
        Fraction
            Duration unit
        """
        sf = self.s.flat.notesAndRests
        unit = Fraction(sf[0].duration.quarterLength)
        for n in sf:
            unit = fraction_gcd(unit, Fraction(n.duration.quarterLength))
        return fraction_gcd(unit, Fraction(1,1)) # make sure 1 is dividable by the unit.denominator

    #return number of ticks per quarter note
    def getResolution(self) -> int:
        """Return the number of ticks per quarter note given the duration unit.

        Returns
        -------
        int
            number of ticks per quarter note.
        """
        unit = self.getDurationUnit()
        #number of ticks is 1 / unit (if that is an integer)
        ticksPerQuarter = unit.denominator / unit.numerator
        if ticksPerQuarter.is_integer():
            return int(unit.denominator / unit.numerator)
        else:
            print(self.s.filePath, ' non integer number of ticks per Quarter')
            return 0

    def getOnsets(self):
        """Returns a list of onsets (ints). Onsets are multiples of the duration unit.

        N.B. this replaces the values of the feature 'onsettick' in the MTC json with the
        newly computed values.

        Returns
        -------
        list of int
            Onset for each note.
        """
        ticksPerQuarter = self.getResolution()
        onsets = [int(n.offset * ticksPerQuarter) for n in self.s.flat.notes]
        #check whether same onsets in songfeatures
        assert len(self.mtcsong['features']['onsettick']) == len(onsets)
        #NB. If initial rests (e.g. NLB142326_01) all onsets are shifted wrt MTC
        # REPLACE MTC values of onsettics with onsets as computed here (anyway)
        for ix in range(len(onsets)):
            self.mtcsong['features']['onsettick'][ix] = onsets[ix]                
        return onsets

    # s : music21 stream
    def removeGrace(self, s):
        """Remove all grace notes from the music21 stream.

        Parameters
        ----------
        s : music21 Stream
            Object representing the score of the song.
        """
        #highest level:
        graceNotes = [n for n in s.recurse().notes if n.duration.isGrace]
        for grace in graceNotes:
            grace.activeSite.remove(grace)
        #if s is not flat, there will be Parts and Measures:
        for p in s.getElementsByClass(m21.stream.Part):
            #Also check for notes at Part level.
            #NLB192154_01 has grace note in Part instead of in a Measure. Might be more.
            graceNotes = [n for n in p.recurse().notes if n.duration.isGrace]
            for grace in graceNotes:
                grace.activeSite.remove(grace)
            for ms in p.getElementsByClass(m21.stream.Measure):
                graceNotes = [n for n in ms.recurse().notes if n.duration.isGrace]
                for grace in graceNotes:
                    grace.activeSite.remove(grace)
        
    # add left padding to partial measure after repeat bar
    def padSplittedBars(self, s):
        """Add padding to bars that originate from splitting a bar at a repeat sign. The second of the two
        resulting (partial) bars should have a padding equal to the lenght of the first (partial) bar in 
        order to obtain correct beatstrength values.

        Parameters
        ----------
        s : music21 Stream
            Object representing the score of the song.

        Returns
        -------
        music21 Stream
            s with padded bars
        """
        partIds = [part.id for part in s.parts] 
        for partId in partIds: 
            measures = list(s.parts[partId].getElementsByClass('Measure')) 
            for m in zip(measures,measures[1:]): 
                if m[0].quarterLength + m[0].paddingLeft + m[1].quarterLength == m[0].barDuration.quarterLength: 
                    m[1].paddingLeft = m[0].quarterLength 
        return s

    #N.B. contrary to the function currently in MTCFeatures (nov 2022), do not flatten the stream
    def parseMelody(self):
        """Converts **kern to music21 Stream and do necessary preprocessing:
        - pad splitted bars
        - strip ties
        - remove grace notes
        The notes in the resulting score correspond 1-to-1 with the notes in MTCFeatures.

        Returns
        -------
        music21 Stream
            music21 score for the song.

        Raises
        ------
        ParseError
            Raised if the **kern file is unparsable.
        """
        try:
            s = m21.converter.parse(self.krnfilename)
        except m21.converter.ConverterException:
            raise ParseError(self.krnfilename)
        #add padding to partial measure caused by repeat bar in middle of measure
        s = self.padSplittedBars(s)
        s = s.stripTies()
        self.removeGrace(s)
        return s

    #Add metric grid
    def create_beatstrength_grid(self):
        """Creates a vector with for each possible onset the beatstrength. The last onset corresponds to the
        end of the last note.

        Returns
        -------
        list (float)
            list with beatstrengths. The beatstrenght for onset n is beatstrength_grid[n].
        """
        beatstrength_grid = []
        unit = Fraction(1, self.getResolution()) #duration of a tick
        s_onsets = m21.converter.parse(self.krnfilename) #original score
        for p in s_onsets.getElementsByClass(m21.stream.Part):
            for m in p.getElementsByClass(m21.stream.Measure):
                offset = 0*unit
                while offset < m.quarterLength:
                    n = m21.note.Note("C").getGrace()
                    m.insert(offset, n)
                    beatstrength_grid.append(n.beatStrength)
                    offset += unit
        assert beatstrength_grid[self.onsets[-1]] == self.mtcsong['features']['beatstrength'][-1]
        return beatstrength_grid

    def add_features(self):
        """Adds a few features that are needed for computing pitch vectors. One value for each note.
        - syncope: True if the note is a syncope (there is a a higher metric weight in the span of the note than at the start of the note).
        - maxbeatstrength: the highest beatstrenght DURING the note.
        - offsets: offset tick of the note (first tick AFTER the note)
        - beatinsong_float: float representation of beatinsong
        - is final pitch: bool. True if the pitch does not change anymore (final pitch might be repeated)
        """
        self.mtcsong['features']['syncope'] = [False] * len(self.mtcsong['features']['pitch'])
        self.mtcsong['features']['maxbeatstrength'] = [0.0] * len(self.mtcsong['features']['pitch'])
        self.mtcsong['features']['beatstrengthgrid'] = self.beatstrength_grid
        beatstrength_grid_np = np.array(self.beatstrength_grid)
        #offsets
        self.mtcsong['features']['offsettick'] = [0]*self.getSongLength()
        ticksperquarter = self.getResolution()
        for ix in range(self.getSongLength()):
            duration = int( Fraction(self.mtcsong['features']['duration_frac'][ix]) * ticksperquarter )
            self.mtcsong['features']['offsettick'][ix] = self.mtcsong['features']['onsettick'][ix] + duration
        #maxbeatstrength
        for ix, span in enumerate(zip(self.onsets,self.mtcsong['features']['offsettick'])): # (onset, offset)
            self.mtcsong['features']['maxbeatstrength'][ix] = self.mtcsong['features']['beatstrength'][ix]
            if np.max(beatstrength_grid_np[span[0]:span[1]]) > self.mtcsong['features']['beatstrength'][ix]:
                self.mtcsong['features']['syncope'][ix] = True
                self.mtcsong['features']['maxbeatstrength'][ix] = np.max(beatstrength_grid_np[span[0]:span[1]])
        #final note:
        #self.mtcsong['features']['maxbeatstrength'][-1] = self.mtcsong['features']['beatstrength'][-1]
        #beatinsong_float
        self.mtcsong['features']['beatinsong_float'] = [float(Fraction(b)) for b in self.mtcsong['features']['beatinsong']]
        #isfinalpitch
        self.mtcsong['features']['isfinalpitch'] = [False] * self.getSongLength()
        finalpitch = self.mtcsong['features']['pitch40'][-1] % 40 #disregard octave
        for ix in reversed(range(self.getSongLength())):
            if self.mtcsong['features']['pitch40'][ix] % 40 == finalpitch:
                self.mtcsong['features']['isfinalpitch'][ix] = True
            else:
                break
        #startfinalpitch
        self.mtcsong['features']['startfinalpitch'] = [False] * self.getSongLength()
        for ix in reversed(range(self.getSongLength()-1)):
            if self.mtcsong['features']['pitch40'][ix] % 40 != finalpitch:
                self.mtcsong['features']['startfinalpitch'][ix+1] = True
                break
        #highestweight: highest weight seen in preceeding notes during the sequence of FINAL pitch only
        self.mtcsong['features']['highestweight'] = [0.0] * self.getSongLength()
        for ix in range(self.getSongLength()):
            if not self.mtcsong['features']['isfinalpitch'][ix]:
                continue
            if self.mtcsong['features']['startfinalpitch'][ix]:
                continue
            #now we are sure that we are 'in' the final tone
            if self.mtcsong['features']['beatstrength'][ix-1] > self.mtcsong['features']['highestweight'][ix-1]:
                self.mtcsong['features']['highestweight'][ix] = self.mtcsong['features']['beatstrength'][ix-1]
            else:
                self.mtcsong['features']['highestweight'][ix] = self.mtcsong['features']['highestweight'][ix-1]


    def getReducedSong(self, ixs_remove, prolong_previous=False):
        """Create a new Song object without the notes in ixs_removed.

        Parameters
        ----------
        ixs_remove : list (int)
            Contains the indices of the notes that need to be removed.
        prolong_previous : bool
            If True, replace the to-be-removed note with a prolongation of the previous note.
            (Because of a bug in music21, this currently does not work if the song contains tuplets.)
            If False, replace the to-be-removed note with a rest.
            (Because of a bug in music21, this currently does not work if the song contains tuplets.)
        
        Returns
        -------
        Song object
            Song object
        """
        #create deep copy of m21 stream and beatstrength_grid
        s_new = copy.deepcopy(self.s)
        mtcsong_new = copy.deepcopy(self.mtcsong)
        beatstrength_grid_new = copy.deepcopy(self.beatstrength_grid)

        ixs_remove = sorted(ixs_remove)

        #Replace notes with prolongation of previous note (with ties, and then stripTies())
        #BUT not when the previous symbol is a rest!
        s_new_list = list(s_new.flat.notes)
        for ix in ixs_remove:

            #for now, leave 0 in. Subsequent notes might have to be removed as well
            if ix == 0:
                continue

            #get note
            n = s_new_list[ix]

            if prolong_previous:
                #check whether previous symbol is a rest
                prev_symbol = n.previous() #seems to work. It takes the right stream.
                while not 'Rest' in prev_symbol.classSet and not 'Note' in prev_symbol.classSet:
                    prev_symbol = prev_symbol.previous()
                if prev_symbol == None:
                    print("Beginning of stream encountered. Should not happen.")
                else:
                    if prev_symbol.isRest:
                        site = n.sites.getSitesByClass('Measure')[0] #assume it is the first (and only)
                        site.remove(n)
                        continue

                #previous symbol is not a rest
                n_prev = s_new_list[ix-1] #exists
                p_new = copy.deepcopy(s_new_list[ix-1].pitch) #take the pitch object of previous note
                s_new_list[ix].pitch = p_new
                #add ties
                if n_prev.tie == None:
                    n_prev.tie = m21.tie.Tie('start')
                else: #previous has tie (must be 'stop')
                    if n_prev.tie.type == 'stop':
                        n_prev.tie = m21.tie.Tie('continue')
                    elif n_prev.tie.type == 'continue':
                        print('Tie of previous note should not be of type continue')
                    else:
                        print('Tie of previous note should not be of type ', n_prev.tie.type)
                n.tie = m21.tie.Tie('stop')
            else: # just remove the note. Will be filled with rests later
                site = n.sites.getSitesByClass('Measure')[0] #assume it is the first (and only)
                site.remove(n)

        #update offsets and durations based on ties:
        #backward, and stop at 1 (TODO)
        if prolong_previous:
            for ix in reversed(range(1, len(s_new_list))):
                n = s_new_list[ix]
                n_prev = s_new_list[ix-1]
                if n.tie != None:
                    if n.tie.type != 'start':
                        #copy offset of tied note to previous note
                        mtcsong_new['features']['offsettick'][ix-1] = mtcsong_new['features']['offsettick'][ix]
                        #add duration to previous note
                        mtcsong_new['features']['duration_frac'][ix-1] = str( Fraction(mtcsong_new['features']['duration_frac'][ix-1]) + Fraction(mtcsong_new['features']['duration_frac'][ix]) )

        #replace removed notes with rests

        for m in s_new.recurse(classFilter=('Measure')):
            m.makeRests(inPlace=True, fillGaps=True, timeRangeFromBarDuration=True)

        #remove ties
        if prolong_previous:
            s_new = s_new.stripTies()

        #now remove first note (if 0 in ixs_remove)
        #also fix in mtcsong_new (below)
        #TODO

        #go over all features in mtcsong_new and remove... this will make a mess
        #for offsettick
        for feat in mtcsong_new['features'].keys():
            for ix in reversed(ixs_remove):
                if ix != 0: #TODO See above
                    mtcsong_new['features'][feat].pop(ix)
        #list of onsets might need to be divided by the gcd
        onsetgcd = np.gcd.reduce(mtcsong_new['features']['onsettick'])
        if onsetgcd > 1:
            for ix in range(len(mtcsong_new['features']['onsettick'])):
                mtcsong_new['features']['onsettick'][ix] = int( mtcsong_new['features']['onsettick'][ix] / onsetgcd )

        song_new = Song(mtcsong_new, None, s_in=s_new, beatstrength_grid_in=beatstrength_grid_new)
        return song_new


    def getColoredSong(self, colordict, lyrics=None, lyrics_ixs=None, title=None):
        """Create a new music21 stream with notes colored according to `colordict`.

        Parameters
        ----------
        colordict : dict
            The keys are the colors, the values the indices of the notes with that color. E.g. {'red':[0,10,11],'grey':[-1]}
            colors notes at indices 0, 10, and 11 red, and the last note grey.
        lyrics : list (str) OR list of lists (str)
            Use the items in this list as lyrics, one string for each note.
            If a list of lists is provided, these will be printed as different lines of lyrics.
        lyrics_ixs : list (int)
            If given, put the lyrics at the given indices in the song. Should have the same length as lyrics.
            If not given, lyrics should have the same length as the song.

        Returns
        -------
        music21 Stream
            music21 Stream.
        """
        if self.reduced:
            s = copy.deepcopy(self.s)
        else:
            s = self.parseMelody()
        if title != None:
            s.metadata.title = title
        #check for right length #if so, assume notes correspond with features
        assert self.getSongLength() == len(s.flat.notes)
        for color, ixs in colordict.items():
            for ix in ixs:
                s.flat.notes[int(ix)].style.color = color
        #add index of note as lyric
        if lyrics == None:
            lyrics = [str(ix) for ix in range(len(s.flat.notes))]
            lyrics_ixs = list(range(len(list(s.flat.notes))))
        if lyrics_ixs is None:
            if len(lyrics) == len(list(s.flat.notes)):
                lyrics_ixs = list(range(len(list(s.flat.notes))))
            else:
                print('Provide indices for the lyrics')
                for ix, n in enumerate(s.flat.notes):
                    n.lyric = None
                return s
        #make sure it is a list (to use .index)
        lyrics_ixs = list(lyrics_ixs)
        #are more lines of lyrics provided?
        multipleLines = False
        if lyrics != None:
            if type(lyrics[0]) == list:
                multipleLines = True
        for ix, n in enumerate(s.flat.notes):
            n.lyric = None
            try:
                lyricix = lyrics_ixs.index(ix)
            except ValueError:
                continue
            if multipleLines:
                for line in lyrics:
                    n.addLyric(line[lyricix])
            else:
                n.addLyric(lyrics[lyricix])
        return s
    
    #we need to repair lily generated by m21 concerning color
    #\override Stem.color -> \once\override Stem.color
    #\override NoteHead.color -> \once\override NoteHead.color

    def repairLyline(self, line):
        """Corrects possbile errors in a line of the Ly export:
        - Note coloring should be done once.
        - Melisma are somehow following beams (instead of slurs)
        - Beaming is wrong. 16th and 32th etc notes get 1 beam.
        
        Parameters
        ----------
        line : str
            a line of a generated lilypond file.

        Returns
        -------
        str
            corrected line
        """
        line = line.replace("\\override Stem.color","\\once\\override Stem.color")
        line = line.replace("\\override NoteHead.color","\\once\\override NoteHead.color")
        line = line.replace("\\include \"lilypond-book-preamble.ly\"","")

        line = line.replace("\\set stemLeftBeamCount = #1", "")        
        line = line.replace("\\set stemRightBeamCount = #1", "")        
        return line
    
    def formatAndRepairLy(self, filename):
        """Go over a lilypond file, and correct the lines (see `self.repairLyline`).
        Clear tagline.
        Set indent of first system to 0.

        Parameters
        ----------
        filename : str
            Filename of the lilypond file to repair.
        """
        with open(filename,'r') as f:
            lines = [self.repairLyline(l) for l in f.readlines()]
        lines = lines + [ f'\paper {{ tagline = "" \nindent=0}}']
        with open(filename,'w') as f:
            f.writelines(lines)

    def insertFilenameLy(self, filename):
        """Insert a header with the filename into a lilypond source file.

        Parameters
        ----------
        filename : str
            Filename of the lilypond file to process.
        """
        with open(filename,'r') as f:
            lines = [l for l in f.readlines()]
        with open(filename,'w') as f:
            for l in lines:
                if "\\score" in l:
                    f.write(f'\\header {{ opus = "{filename}" }}\n\n')
                f.write(l)

    def insertSystemSpacingLy(self, filename):
        """Insert a paper block with system-system spacing into lilypond source file.

        Parameters
        ----------
        filename : str
            Filename of the lilypond file to process.
        """
        with open(filename,'r') as f:
            lines = [l for l in f.readlines()]
        with open(filename,'w') as f:
            for l in lines:
                f.write(l)
                if "\\version" in l:
                    f.write( "\paper { system-system-spacing.basic-distance = #16 }\n" )

    def createColoredPDF(self, colordict, outputpath, filebasename=None, showfilename=True, lyrics=None, lyrics_ixs=None, title=None):
        """Create a pdf with a score with colored notes.

        Parameters
        ----------
        colordict : dict
            The keys are the colors, the values the indices of the notes with that color. E.g. {'red':[0,10,11],'grey':[-1]}
            colors notes at indices 0, 10, and 11 red, and the last note grey.
        outputpath : str
            name of the output directory
        filebasename : str, default None
            basename of the pdf file to generate (without .pdf). If None, the identifier of the song as provided by
            MTCFeatures is used as file name.
        showfilename : bool, default True
            Include the filename in the pdf (lilypond opus header).

        Returns
        -------
        path-like object
            Full path of the generated pdf.
        """
        if filebasename == None:
            filebasename = self.mtcsong['id']
        s = self.getColoredSong(colordict, lyrics=lyrics, lyrics_ixs=lyrics_ixs, title=title)
        s.write('lily', os.path.join(outputpath, filebasename+'.ly'))
        self.formatAndRepairLy(os.path.join(outputpath, filebasename+'.ly'))
        if showfilename:
            self.insertFilenameLy(os.path.join(outputpath, filebasename+'.ly'))
        self.insertSystemSpacingLy(os.path.join(outputpath, filebasename+'.ly'))
        output = subprocess.run(["lilypond", os.path.join(outputpath, filebasename+'.ly')], cwd=outputpath, capture_output=True)
        return os.path.join(outputpath, filebasename+'.pdf')

    def createColoredPNG(self, colordict, outputpath, filebasename=None, showfilename=True, lyrics=None, lyrics_ixs=None, title=None):
        """Create a png with a score with colored notes.

        Parameters
        ----------
        colordict : dict
            The keys are the colors, the values the indices of the notes with that color. E.g. {'red':[0,10,11],'grey':[-1]}
            colors notes at indices 0, 10, and 11 red, and the last note grey.
        outputpath : str
            name of the output directory
        filebasename : str, default None
            basename of the png file to generate (without .png). If None, the identifier of the song as provided by
            MTCFeatures is used as file name.
        showfilename : bool, default True
            Include the filename in the png (lilypond opus header).

        Returns
        -------
        path-like object
            Full path of the generated png.
        """
        pdf_fn = self.createColoredPDF(colordict, outputpath, filebasename, showfilename, lyrics=lyrics, lyrics_ixs=lyrics_ixs, title=title)
        png_fn = pdf_fn.replace('.pdf','.png')
        output = subprocess.run(['convert', '-density', '100', pdf_fn, '-alpha', 'Remove', '-trim', png_fn], cwd=outputpath, capture_output=True)
        return png_fn
    
    def showColoredPNG(self, colordict, outputpath, filebasename=None, showfilename=True, lyrics=None, lyrics_ixs=None, title=None):
        """Show a png with a score with colored notes. For use in a Jupyter notebook.

        Parameters
        ----------
        colordict : dict
            The keys are the colors, the values the indices of the notes with that color. E.g. {'red':[0,10,11],'grey':[-1]}
            colors notes at indices 0, 10, and 11 red, and the last note grey.
        outputpath : str
            name of the output directory
        filebasename : str, default None
            basename of the png file to generate (without .png). If None, the identifier of the song as provided by
            MTCFeatures is used as file name.
        showfilename : bool, default True
            Include the filename in the png (lilypond opus header).
        """
        png_fn = self.createColoredPNG(colordict, outputpath, filebasename, showfilename, lyrics=lyrics, lyrics_ixs=lyrics_ixs, title=title)
        display.display(display.Image(png_fn))

    def showPNG(self, lyrics=None, lyrics_ixs=None):
        """Show a png with a score of the song. For use in a Jupyter notebook.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.showColoredPNG({}, tmpdirname, showfilename=False, lyrics=lyrics, lyrics_ixs=lyrics_ixs)
    
    def createPNG(self, outputpath, filebasename=None, showfilename=False, lyrics=None, lyrics_ixs=None, title=None):
        return self.createColoredPNG({}, outputpath, filebasename=filebasename, showfilename=showfilename, lyrics=lyrics, lyrics_ixs=lyrics_ixs, title=title)


    def writeMTCJSON(self, outputpath, filebasename=None):
        if filebasename == None:
            filebasename = self.mtcsong['id']        
        with open(os.path.join(outputpath,filebasename+'.json'), 'w') as f:
            json.dump(self.mtcsong, f)
    
    def writePerSymbolJSON(self, outputpath, filebasename=None):
        if filebasename == None:
            filebasename = self.mtcsong['id']
        #build per symbol representation
        symbols = []
        for ix in range(self.getSongLength()):
            symbol = {}
            #due to a bug in MTCFeatures, there is no octave information in the pitch40 feature
            #figure out the octave from the midi pitch, and correct the pitch40 value
            #take 1-based pitch40. I.e., the first octave is [1,40] (rather than [0,39])
            #midi middle C octave = 5 (sixth octave) = base40 octave 4 (fourth octave)
            octave = self.mtcsong['features']['midipitch'][ix] // 12
            pitch40 = 40 * (octave-2) + ((self.mtcsong['features']['pitch40'][ix] + 1) % 40) - 1  #still do mapping to base octave for pitch40 first, to make this future-proof.
            symbol["pitch40"] = pitch40
            symbol["onset"] = self.mtcsong['features']['onsettick'][ix]
            symbol["phrase"] = self.mtcsong['features']['phrase_ix'][ix]
            symbol["ima"] = self.mtcsong['features']['imaweight'][ix]
            symbol["phrasepos"] = self.mtcsong['features']['phrasepos'][ix]
            symbol["beatstrength"] = self.mtcsong['features']['beatstrength'][ix]
            symbols.append(symbol)
        songdictINNER = {}
        songdictINNER['symbols'] = symbols
        songdict = {}
        songdict[filebasename] = songdictINNER
        with open(os.path.join(outputpath,filebasename+'.json'), 'w') as f:
            json.dump(songdict, f)