from dataclasses import dataclass, field
import numpy as np



# note class stores equivalent note names and correpsonding frequency in Hz
@dataclass(order=True)
class Note:
    frequency: float = 0
    name: str = field(default_factory=str)
    sharp_notation: str = field(default_factory=str)
    flat_notation: str = field(default_factory=str)

    def __post_init__(self):

        flats = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
        sharps = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

        octave = self.name[-1]

        # initialize notations from self.name
        if '#' in self.name:
            index = sharps.index(self.name[:-1])
            self.sharp_notation = self.name
            self.flat_notation = flats[index] + octave
        elif 'b' in self.name:
            index = flats.index(self.name[:-1])
            self.flat_notation = self.name
            self.sharp_notation = sharps[index] + octave
        else:
            index = flats.index(self.name[:-1])
            self.sharp_notation = self.name
            self.flat_notation = self.name

        # initialize frequency in Hz
        number_from_C0 = 12 * int(octave) + index
        self.frequency = 440 * (2**(1/12))**(number_from_C0-57)



class Chord:
    def __init__(self, notes: tuple):
        self.notes = notes
        self.semitones = []
        self.intervals = []
        self.chord = self.notes[0].name[:-1]
        self.indexes = []

        # look up tables
        self.flats = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]

        self.interval_names = ['unison','min 2nd','maj 2nd','min 3rd','maj 3rd','prf 4th',       # indexes 0 - 5
                               'tritone','prf 5th','min 6th','maj 6th','dom 7th','maj 7th',      # indexes 6 - 11
                               'octave','min 9th','maj 9th','min 3rd','maj 3rd','prf 11th',      # indexes 12 - 17
                               'tritone','prf 5th','min 13th','maj 13th','min 7th','maj 7th']    # indexes 18 - 23

        # maps to self.interval names
        chord_type = ['',''    ,'sus2','min','maj','sus4' ,'b5',''     ,'','','dom','maj7',
                      '','min9','maj9','min','maj','add11',''  ,'add13','','',''   ,'maj7']
        
        # get indexes of note names using flats list and flat_notation from Note
        for note in self.notes:
            self.indexes.append(self.flats.index(note.flat_notation[:-1]))

        # get semitones from chord root
        for index in range(len(self.indexes) - 1):
            diff = self.indexes[index + 1] - self.indexes[0]
            if diff < 0:
                diff = diff + len(self.flats)
            self.semitones.append(diff)

        # semitones map to interval_names
        for num in self.semitones:
            self.intervals.append(self.interval_names[num])

        # determine chord type
        for num in self.semitones:
            # make sure not to repeat chord type names
            isrepeat = chord_type[num][:3]
            if isrepeat in self.chord:
                self.chord = self.chord.replace(isrepeat,chord_type[num])
            else:
                self.chord = self.chord + chord_type[num]

    # print values
    def __repr__(self):
        return '{}: {}, {}'.format(self.__class__.__name__, self.chord, self.intervals)

    def get_intervals_btwn_each_note(self):
        interval_indexes = []
        intervals = []
        for index in range(len(self.indexes) - 1):
            diff = self.indexes[index + 1] - self.indexes[index]
            if diff < 0:
                diff = diff + len(self.flats)
            interval_indexes.append(diff)
            intervals.append(self.interval_names[diff])
        print('Intervals between each note in the chord:')
        return intervals



class Scale:
    def __init__(self, root: Note, scale = 'major'):
        self.root = root
        self.scale_name = scale
        self.scale = [self.root]
        octave = self.root.name[-1]

        # scales are listed by step sizes in semitones
        scales = {
            'major':      [2,2,1,2,2,2,1],
            'dimW':       [2,1,2,1,2,1,2,1],
            'dimH':       [1,2,1,2,1,2,1,2],
            'pentmaj':    [2,2,3,2,3],
            'bebopmaj':   [2,2,1,2,1,1,2,1],
            'harmmaj':    [2,2,1,2,1,3,1],
            'lydaug':     [2,2,2,2,1,2,1],
            'augment':    [3,1,3,1,3,1],
            'blues':      [3,2,1,1,3,2]
        }
        # generate modes from major, stored in scales
        modes = ['ionian','dorian','phrygian','lydian',
                'mixolydian','aeolian','locrian']
        for k in range(6):
            scales[modes[k+1]] = np.roll(scales['major'],-(k+1))

        # set semitones from scale arg
        self.semitones = scales[self.scale_name]

        # notation references
        self.flats = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
        self.sharps = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

        if self.root.name[:-1] == 'F' or 'b' in self.root.name[:-1]:
            notation = self.flats
        else:
            notation = self.sharps

        #scale_notes = [self.root.name]
        index = notation.index(self.root.name[:-1])
        for num in self.semitones:
            index = (index + num) % 12
            note = Note(name=(notation[index] + octave))
            self.scale.append(note)
            #scale_notes.append(notation[index] + octave)
            if index == 11:
                octave = str(int(octave) + 1)

    def __repr__(self) -> str:
        return '{} {} {}'.format(self.root.name, self.scale_name, [self.scale[x].name for x in range(len(self.scale))])

    def make_chord(self):
        return Chord(self.scale[::2])



if __name__ == '__main__':

    # list of note names as strings
    #chord1 = ['G4','B4','D5']
    chord1 = ['G4','B4','C4','D5','F#5']

    # create note object for each note name, save as tuple
    chord_tones = []
    for note in chord1:
        chord_tones.append(Note(name=note))
    chord1 = tuple(note for note in chord_tones)

    # chord object currently only works with major triad
    gmaj = Chord(chord1)

    '''
    print(gmaj)
    for note in gmaj.notes:
        print(note)
    print(gmaj.get_intervals_btwn_each_note())
    '''

    gmajscale = Scale(Note(name='G4'), scale='lydian')
    print(gmajscale)
    #print(gmajscale.make_chord())

    # create a list of the circle of fifths
    # notation references
    flats = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
    sharps = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    circle_fifths = []
    index = 0
    for num in range(len(sharps)):
        if num > 6:
            circle_fifths.append(Note(name=flats[index] + '4'))
        else:
            circle_fifths.append(Note(name=sharps[index] + '4'))
        index = (index + 7) % 12
    print([x.name for x in circle_fifths])
