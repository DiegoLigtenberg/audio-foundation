
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

random.seed(42) 
np.random.seed(42)  # Replace 42 with any integer seed of your choice

music_genres = [
    "ambient tech", "acoustic goth", "afro-experimental", "ambient house", "acid jazz fusion", 
    "avant-garde pop", "bluegrass metal", "baroque punk", "bubblegum pop", "bohemian electronic", 
    "chamber funk", "chiptune jazz", "cyberpunk industrial", "crunk jazz", "deep trance", 
    "doomstep", "drone folk", "dark wave", "dystopian synthwave", "electro-swing", 
    "electro-blues", "flamenco trap", "folkstep", "future garage", "funky progressive", 
    "gothic metalcore", "grimehouse", "groovy house", "hard trance", "indie jazzcore", 
    "jungle hip-hop", "k-pop metal", "laserwave", "latin soul", "lo-fi ambient", "minimal punk", 
    "neo-psychadelic", "neurofunk", "new age jazz", "noir synthwave", "post-dubstep", 
    "progressive electro", "psytrance industrial", "punk-hop", "retro-futuristic", "ska-metal", 
    "shamanic techno", "sludge jazz", "soulful trance", "space disco", "stomp blues", "swampy rock", 
    "tango techno", "tribal house", "vaporwave jazz", "viking metalcore", "visual kei fusion", 
    "witch house", "zombie folk", "urban bluegrass", "acid soul", "alt-electronic", 
    "ambient trap", "analog glitch", "baroque hip-hop", "bass synthwave", "blaxploitation funk", 
    "boogie-down house", "breakbeat soul", "brostep", "bubblegum electro", "chillwave rock", 
    "cloud rap", "comeback R&B", "cool jazzcore", "cosmic disco", "crunk rock", "cumbia dub", 
    "cyber metal", "dark electronic", "deathstep", "drift rock", "dubpunk", "dreamwave", 
    "drillstep", "electroclash", "electropunk", "electronic rock", "feminist punk", 
    "futuristic jazz", "ghetto funk", "gothic trance", "groove metal", "hardcore funk", 
    "haunted house", "heavy dub", "horror jazz", "industrial funk", "indie techno", "jazzcore", 
    "juke-house", "kawaii pop", "kraut metal", "lazer jazz", "melodic doom", "metalcore jazz", 
    "mid-century jazz", "minimal rock", "noise-hop", "nu disco jazz", "occasional trance", 
    "one-man-band metal", "psychedelic opera", "punk jazz fusion", "punkwave", "savant pop", 
    "saxophone house", "sci-fi jazz", "screamo rock", "shred-hop", "sincere soul", 
    "space opera", "speedcore blues", "stomp techno", "surf-hop", "swamp trance", "trap jazz", 
    "tropical dubstep", "tropical synthwave", "twee goth", "vapor trap", "vintage synthpop", 
    "warped metal", "witch-step", "art punk", "barbershop swing", "bossa nova house", "bounce funk", 
    "broken beat soul", "calypso swing", "caribbean punk", "chillbient", "clashstep", 
    "clique synthwave", "club rap", "conceptual jazz", "cyber brass", "darkwave pop", 
    "dancehall jazz", "dark electro", "dirty synthwave", "doom metalcore", "drumstep", 
    "electro-funk", "electronic classical", "electropop metal", "experimental swing", 
    "field-recording jazz", "flute funk", "freak-folk", "funk-metal fusion", "grindstep", 
    "guitar funk", "hard rock synthwave", "highlife electronica", "horrorcore", "house metal", 
    "indie electronic rock", "jazz rock metal", "jungle drum and bass", "kawaii house", 
    "krautrock pop", "leisurewave", "lounge punk", "math-hop", "minimal funk", "minimal jazz", 
    "minimalist experimental", "modular synthwave", "narrative pop", "noir jazz", "outlaw electronica", 
    "polyrhythm funk", "post-apocalyptic rock", "psy-dub", "psy-electronica", "punkwave", "soulstep", 
    "spiritual house", "steampunk blues", "stoner-surf", "techno jazz fusion", "tropical reggae", 
    "trumpet jazz", "twist-hop", "urban experimental", "vintage electro", "western metal", 
    "yacht rock fusion", "yoga trance", "zydeco synthwave", "ambient soul", "avant-garde folk", 
    "balkan metal", "baroque dubstep", "bass clarinet jazz", "beach house", "bluegrass blues", 
    "blues grunge", "blues rock fusion", "bohemian electro", "bubblegum rock", "campfire punk", 
    "chamber pop", "chillstep rock", "clothesline indie", "country-wave", "cowpunk", "crossover jazz", 
    "dark folk", "death jazz", "death techno", "dreamy rock", "drum n bass jazz", "drums-only jazz", 
    "electro-ambient", "electronic swing", "electro-blues", "folk fusion", "folktronica", 
    "future funk rock", "fuzz jazz", "gamelan techno", "ghetto jazz", "guitar ambient", 
    "hardcore trance", "hip-hop jazz", "indie classical", "indie metal", "indie trance", 
    "industrial grunge", "intergalactic jazz", "jazz blues fusion", "jazzcore rock", 
    "jungle dubstep", "lo-fi synthwave", "lounge jazz", "mash-up dubstep", "minimal post-punk", 
    "new age blues", "neo-gothic jazz", "noir trance", "old school rock", "orchestral dubstep", 
    "pop-jazz fusion", "post-apocalyptic blues", "post-punk jazz", "progressive blues", "psychedelic rock", 
    "punk funk", "punk jazz", "psytrance blues", "reggae-funk", "salsa jazz", "savant jazz", 
    "screamo-metal", "soul-metal fusion", "space jazz", "surf punk", "tech-house", "techno blues", 
    "trance metal", "trap-funk", "urban folk", "vaporwave punk", "weird rock", "west coast jazz", 
    "yodeling rock", "swing-dance fusion", "electrogrunge", "grunge jazz", "urban jazz fusion", 
    "futurepop", "space rock", "swamp jazz", "new wave jazz", "arabic folk", "north african jazz", 
    "arabian trance", "sufi jazz", "sikh ambient", "pan-pacific jazz", "taiwanese blues", 
    "latino-electro", "latin synthwave", "peruvian jazz", "cuban trap", "mexican hip hop", 
    "dutch psychedelic", "uk hardcore", "british jazz fusion", "nordic black metal", 
    "scandinavian jazz", "finnish experimental", "portuguese jazz", "brazilian metal", 
    "australian blues", "afrobeat house", "new zealand dub", "jamaican dub", "french dubstep", 
    "german electro", "indonesian indie", "american experimental", "italian house", 
    "spanish flamenco fusion", "czech jazz", "german krautrock", "belgian electro pop", 
    "chilean indie", "polish jazz metal", "swiss ambient", "norwegian folk rock", "french electro swing", 
    "australian soul", "new york rap", "baltimore club", "portuguese punk", "american avant-garde jazz", 
    "barcelona street jazz", "indie-folk pop", "berlin house", "argentine rock", "london techno", 
    "turkish pop", "dutch synthwave", "indie rock dub", "new york hip-hop", "indian rock", 
    "african chillout", "vancouver trance", "south african jazz", "scottish soul", "swedish techno", 
    "sydney hip-hop", "copenhagen electro", "greek psychedelic", "vancouver blues", "belgium bass", 
    "singaporean jazz", "icelandic electro", "bangladeshi indie", "danish chillout", "middle eastern jazz", 
    "indonesian jazz", "moroccan electronica", "japanese grunge", "kenyan reggae", "portuguese funk", 
    "colombian folk", "mauritian jazz", "kenyan house", "cambodian funk", "french indie pop", 
    "nigerian afro-fusion", "kenyan hip-hop", "somali funk", "ethiopian jazz", "taiwanese indie", 
    "pakistani rock", "chinese reggae", "mongolian jazz", "french ambient", "new zealand blues", 
    "russian rock", "polish experimental", "armenian soul", "indian dubstep", "filipino blues", 
    "vietnamese jazz", "ukrainian electronic", "brazilian pop", "puerto rican reggae", "czech folk", 
    "thailand indie", "singapore jazz", "kenyan blues", "south african electronic", "indian trap",
]


music_genres.extend([
    "acid rock", "advanced dubstep", "alternative metal", "ambient industrial", "angry jazz", 
    "anarcho-punk", "art pop", "avant-garde metal", "balkan techno", "baroque techno", 
    "batcave rock", "beach wave", "big beat", "blackgaze", "blues rock", "bohemian house", 
    "boogie funk", "booty bass", "bubblegum wave", "calypso house", "chamber electronica", 
    "chillout jazz", "chiptune punk", "clowncore", "circuit bending", "classical trap", 
    "crossover thrash", "cumbia funk", "dark jazz", "dark metal", "dark pop", "dark techno", 
    "deathcore jazz", "dirty south rap", "doom funk", "doom jazz", "drone metal", "dream punk", 
    "drum and bass reggae", "dub-hop", "electro-post-punk", "electropunk disco", "experimental rap", 
    "fablecore", "faux jazz", "fifth-wave ska", "folk metal", "funk jazz", "fusion house", 
    "garage blues", "ghettotech", "glitch hop", "gothic doom", "grindcore jazz", "groovy techno", 
    "guitar-based ambient", "hardboiled jazz", "hardcore electronic", "hardstyle reggae", "harmonic rock", 
    "hip-hop soul", "highlife blues", "hip-hop jazz fusion", "indie post-punk", "industrial dance", 
    "indie folk pop", "industrial doom", "industrial pop", "intelligent dance music", "jangle punk", 
    "jazz-electro fusion", "jazzstep", "japanese synthwave", "jazz fusion blues", "jungle jazz", 
    "k-pop fusion", "krautfolk", "latin jazz fusion", "lo-fi pop", "lo-fi reggae", "looping music", 
    "mashup trance", "melodic death metal", "metal blues", "minimal house", "minimal techno", 
    "mood rock", "morning jazz", "neo-folk", "neo-psychedelia", "neo-soul funk", "new age metal", 
    "new wave soul", "noise rock", "noise synth", "nordic rock", "nu-jazz", "oceanic pop", 
    "old school techno", "operatic rock", "outlaw country", "pagan metal", "polka punk", 
    "post-apocalyptic folk", "post-grunge", "post-metal jazz", "post-punk blues", "psybreaks", 
    "psychedelic dance", "psychedelic funk", "psychedelic indie", "psytrance fusion", "punk jazz fusion", 
    "progressive blues", "progressive metal", "progressive punk", "psy-synth", "powerpop punk", 
    "power metal jazz", "pure trance", "qawwali fusion", "queer pop", "quickstep", "radiohead jazz", 
    "rap jazz fusion", "rockabilly funk", "rockwave", "romantic jazz", "saltwater rock", 
    "screamo-synth", "singer-songwriter jazz", "soulstep", "space funk", "space opera rock", 
    "spacepunk", "spaghetti western pop", "spiritual jazz", "spiritual techno", "stark techno", 
    "stomper techno", "stoner ambient", "stoner dub", "stoner hip-hop", "surf jazz", "swing fusion", 
    "synthwave jazz", "synthpunk", "synth rock", "synth pop jazz", "synthcore", "tech jazz", 
    "techno metal", "techno pop", "techno rock", "techno trance", "thrash blues", "thrash jazz", 
    "thrash metal", "thrash punk", "tropical techno", "trap jazz", "tropical bass", "tropical funk", 
    "twee indie", "twin peaks indie", "urban jazz", "vaporwave ambient", "vaporwave indie", 
    "vaporwave metal", "vaporwave rock", "viking jazz", "viking metal", "visual rock", "vocal jazz", 
    "waltz electro", "west coast synthwave", "witch trance", "witch-house punk", "yacht pop", 
    "yasss trap", "zombie metal", "industrial synthwave", "acid pop", "mathcore", "jungle techno", 
    "psygrind", "eclectic techno", "classical punk", "chillwave pop", "fusion jazz", "space doom", 
    "post-grunge indie", "hardcore jazz", "glitchy synthpop", "noir electro", "dreamsynth", 
    "minimal blues", "melodic grindcore", "speed punk", "salsa rock", "bossa fusion", 
    "asian pop punk", "art jazz", "bluescore", "ambient folk", "avant-ambient", "acid folk", 
    "baroque metal", "baroque blues", "rural jazz", "calypso jazz", "country-electronic fusion", 
    "cloud rap jazz", "crunchy jazz", "cyberpunk jazz", "dark psych-folk", "dystopian jazz", 
    "french-electro fusion", "indie-folk fusion", "indie-rave", "neoclassical ambient", 
    "neoclassical techno", "new soul", "percussive jazz", "post-modern punk", "space jazz", 
    "spiritual folk", "surf-psych", "techno-ballad", "techno-blues", "progressive rock jazz", 
    "chillstep funk", "disco jazz", "synth jazz fusion", "symphonic jazz", "pop-synth", 
    "rock fusion", "russian techno", "surf metal", "percussive trance", "dream trance", 
    "soul techno", "romantic rock", "screamo soul", "future gospel", "vintage dancehall", 
    "electric boogaloo", "romantic folk", "urban techno", "groovy jazz", "electro-blues fusion", 
    "darkwave pop", "dark industrial techno", "grunge fusion", "high-energy trance", "nu-vintage jazz", 
    "percussive synth", "revolutionary jazz", "electrofolk", "supernatural synth", "trap metal", 
    "lo-fi synthpop", "spiritual trap", "soul-hop", "indie-punk fusion", "vintage psychedelic", 
    "acid rock fusion", "dream-folk", "prog-folk", "swamp jazz", "psychedelic bossa", "indie-folk jazz", 
    "percussive ambient", "disco-rock", "trap reggae", "speedcore trance", "hardwave", "housecore", 
    "rockstep", "ethereal techno", "space-funk", "ambient metal", "flamenco punk", "hardwave trance", 
    "progressive electrofunk", "darkwave synthpop", "indie-techno", "dark jazz fusion", 
    "future soul-funk", "neo-synthpop", "psytrance core", "chamber rock", "avant-ambient techno", 
    "new-age psychedelic", "outlaw dance", "extreme ambient", "futuristic jazzcore", "hardcore hip-hop", 
    "indie metalcore", "neurofunk trance", "crossover fusion", "bass funk", "speed garage", 
    "dystopian dubstep", "glitchcore", "dream-fusion", "symphonic ambient", "industrial futurepop", 
    "post-punk soul", "jazz-hop", "cybercore", "breakcore folk", "glitchwave", "indie-electro fusion", 
    "swamp rock", "baroque electro", "highlife electro", "jungle metal", "avant-garde reggae", 
    "screamo-swing", "space-pop", "punk-soul", "glitch synthpop", "tribal-electro", "cyberpunk soul", 
    "dreamcore", "futuristic jazz funk", "dream jazz", "acid house", "avant-synth", "hardcore folk", 
    "grunge-fusion", "trap jazz fusion", "future jazz-funk", "robotic house", "space-metal", 
    "goth-punk", "synthetic indie", "new soul jazz", "synth-grunge", "future funk rock", "cyberpunk rap", 
    "electrofolk funk", "post-industrial jazz", "neo-experimental", "eurobeat metal", "indie techno-folk", 
    "neuro-folk", "cyber dub", "billy-electro", "drumcore jazz", "grunge ambient", "hybrid metal", 
    "noisehouse", "dark dream pop", "horror pop", "mid-tempo trance", "dream-grindcore", "tech house-metal"
])


music_genres.extend([
    "ambient doom", "acid trance", "experimental punk", "cosmic jazz", "psywave", 
    "retro-futuristic rock", "neon jazz", "robot funk", "space doom", "baroque synth", 
    "punk funk", "lo-fi punk", "electro-folk", "tribal metal", "industrial folk", 
    "industrial funk", "glitch rap", "ambient black metal", "psychedelic glitch", 
    "grunge jazz", "drone house", "dreamwave punk", "dreamcore jazz", "psychedelic soul", 
    "gothic rap", "techno-folk fusion", "trap jazz fusion", "high-tech jazz", "jazzgaze", 
    "neo-metal", "future-folk", "metal-hop", "cyberpunk ambient", "chaoscore", "dark chillwave", 
    "shimmer pop", "indie-synth", "dreamcore ambient", "indie-jazz", "space-funk rock", 
    "experimental disco", "glitch-hop jazz", "ambient blues", "new wave funk", "surreal synth", 
    "ambient dancehall", "synthwave punk", "drum-step", "cyberfolk", "post-industrial soul", 
    "post-rock funk", "dark trance", "space jazz fusion", "hyper-pop punk", "glitch disco", 
    "minimal techno funk", "future doom", "math pop", "ambient groove", "gothic swing", 
    "new-age rock", "urban jazz fusion", "doom jazz", "pop-punk electronica", "lo-fi metal", 
    "bossa nova punk", "acid jazz fusion", "punkstep", "futuristic blues", "jazz drone", 
    "dreamstep", "dream soul", "soultrap", "techno country", "space swing", "pagan trance", 
    "lo-fi experimental", "space opera jazz", "metal pop", "dub-noir", "ambient techno", 
    "avant-garde jazz", "post-wave", "tribal punk", "heavy-wave", "glitch industrial", 
    "neon-folk", "underground trap", "techno funk", "tribal techno", "metal techno", 
    "neo-soul electronic", "speed-rap", "space opera", "guitar-based electronic", "stomp blues", 
    "space industrial", "chill-metal", "bluescore ambient", "synth-punk", "psy-pop", "trancecore", 
    "post-metal funk", "neoclassical synthwave", "glitch-soul", "ambient dubstep", "high-tech jazz", 
    "psychedelic orchestral", "future-wave", "outlaw-folk", "cinematic trap", "robotic dub", 
    "acid house jazz", "lo-fi jazzcore", "sludge funk", "future-trap", "psych-funk", "classic-pop punk", 
    "highlife jazz", "lo-fi fusion", "metal swing", "baroque funk", "dreamy indie", "trapstep", 
    "folk doom", "indie-funk", "soul-gaze", "glitchwave", "heavy-synth", "dark experimental pop", 
    "bass-gaze", "dark indie pop", "rock-dub", "new-wave jazz", "chill-hop fusion", "space jazz-funk", 
    "acid-rock jazz", "synth-funk fusion", "shimmer-wave", "hyper-trap", "vaporwave-folk", "drum-break", 
    "trap-industrial", "experimental soul", "space-electronic", "classic synth-wave", "post-punk metal", 
    "classic funk", "ambient metal-pop", "new-wave punk", "grunge-folk", "pop-blues", "ambient post-punk", 
    "tribal-dub", "experimental techno", "dark trance rock", "cyber-pop", "cyber-trap", "lo-fi electronica", 
    "indie-hop", "future-funk blues", "future trap-folk", "punk-trance", "hip-hop metal", "soul-jazz fusion", 
    "dream-jazz", "chill-funk", "cosmic techno", "dreamcore indie", "experimental rap-rock", "new-wave blues", 
    "synth-metal", "post-dancehall", "ambient house jazz", "psy-soul", "acid techno", "goth-folk", 
    "post-punk jazz", "lo-fi soul", "space-blues", "neoclassical pop", "freak-folk", "glitchwave jazz", 
    "space-punk soul", "dark-folk", "freakcore", "electro-metal", "psyambient", "new-wave synthpop", 
    "ambient chillout", "techno-dub", "punk ambient", "space-noise", "darkwave-electro", "chillwave-funk", 
    "swamp punk", "swamp jazz", "progressive-funk", "soulwave", "electronic folk", "grindcore blues", 
    "jungle pop", "chillhop blues", "gothic folk", "indie soul", "lo-fi jazz-pop", "funk-hop", "dark trance jazz", 
    "classical-electro", "post-synth", "speedcore pop", "post-punk trance", "ambient folk", "glitch-rock", 
    "blues-punk", "metal-step", "indie-folk funk", "electro-doom", "grungewave", "melodic doom", "ambient garage", 
    "goth-metal", "space-doom", "grunge-trance", "metal-trap", "synth-soul", "chill-grind", "pop-dub", 
    "techno-punk", "metallic trance", "gothic-ambient", "tribal-funk", "space-soul", "folk-noir", 
    "new-wave electronica", "glitch-step", "jazz-funk-fusion", "dreamy jazz-folk", "avant-garde blues", 
    "rock-noir", "futuristic metal", "new-age funk", "synth-folk", "post-metal synthwave", "dark-blues", 
    "pop-metal", "techno-folk fusion", "metal-soul", "drum-n-bass fusion", "acid-wave", "future-folk funk", 
    "progressive-soul", "avant-garde pop", "grunge-punk", "psy-funk", "ambient-post rock", "space-metal funk", 
    "ambient doomcore", "cyber-folk fusion", "retro-futuristic jazz", "synth-wave punk", "punk-metal", 
    "darkwave funk", "grunge-soul", "hip-hop-funk fusion", "lo-fi metal-pop", "goth-punk fusion", 
    "glitch-metal", "synth-grunge", "post-industrial synthwave", "drone-trap", "post-rock-dub", "bluespunk", 
    "traprock", "baroque-synth", "ambient-soul", "hard-techno jazz", "pop-funk rock", "psy-synth rock", 
    "glitch-soul", "indie-metal", "future-techno-funk", "space-folk jazz", "gothic electro-pop", 
    "experimental-hip hop", "surf-soul", "vaporwave-funk", "ambient-funk", "indie-jazz-rock", "lo-fi jazz-soul"
])

music_genres.extend([
    # Classical & Orchestral
    "baroque", "romantic", "impressionist", "minimalist", "avant-garde classical", "symphonic metal", "chamber music",
    "opera", "sacred choral", "cantata", "film score", "symphonic jazz", "modern classical", "neoclassical", "flamenco",
    
    # Traditional & Folk
    "bluegrass", "Celtic", "Indian classical", "Cajun", "Zulu folk", "Sufi music", "Mongolian throat singing", "Gamelan",
    "Carnatic", "fado", "Arabic maqam", "Tuvan", "Tango", "Haitian Vodou", "Russian folk", "Afrobeat", "Balkan brass band",
    "Highlife", "Reggae", "Samba", "Klezmer", "Polka", "Tongan folk", "Qawwali", "Mariachi", "Griot", "Pagan folk",
    
    # Rock & Alternative
    "psychedelic rock", "grunge", "indie rock", "post-punk", "glam rock", "garage rock", "alt-country", "progressive rock",
    "shoegaze", "stoner rock", "noise rock", "hardcore punk", "emo", "math rock", "post-rock", "experimental rock",
    "dark wave", "crust punk", "psychedelic doom", "sludge metal", "new wave", "post-hardcore", "nu-metal", "indie pop",
    "indie electronic", "punk blues", "metalcore", "death metal", "deathcore", "black metal", "thrash metal", "grindcore",
    
    # Jazz & Blues
    "bebop", "smooth jazz", "soul jazz", "jazz fusion", "free jazz", "cool jazz", "jazz rap", "Moorish jazz", "acid jazz",
    "Delta blues", "Chicago blues", "Texas blues", "electric blues", "classic blues", "jump blues", "swing", "ragtime",
    "Gypsy jazz", "blues rock", "soul", "gospel blues", "Afro-Cuban jazz", "Mambo", "Bossa nova", "Tango jazz", "salsa",
    
    # Electronic & Dance
    "ambient", "drone", "industrial", "minimal techno", "dubstep", "trap", "psytrance", "deep house", "future bass",
    "progressive house", "hardstyle", "tech house", "chiptune", "lo-fi hip hop", "glitch hop", "psy-bient", "techno",
    "UK garage", "dub", "dancehall", "synthwave", "retro wave", "acid house", "vaporwave", "hyperpop", "nu-disco",
    "hardcore techno", "electro swing", "electropop", "cyberpunk", "Chicago house", "drum and bass", "breakbeat",
    
    # World & Regional
    "Celtic folk", "Afro-pop", "Latin jazz", "K-pop", "C-pop", "J-pop", "Bossa nova", "Cumbia", "Reggaeton", "Soca",
    "Dholak", "Mariachi", "Flamenco fusion", "Middle Eastern traditional", "Mugham", "Funk carioca", "Baile funk",
    "Desi hip hop", "Tropicalia", "Tango fusion", "Arabic jazz", "Amapiano", "Zouk", "Chutney music", "Celtic punk",
    
    # Fusion & Hybrid Genres
    "psychobilly", "folk metal", "gypsy punk", "synth-folk", "psychelectric", "ambient techno", "bluegrass jazz",
    "metalcore blues", "punk jazz", "reggae metal", "salsa electronic", "trap soul", "synth-punk", "electronic folk",
    "future jazz", "jazz metal", "indie soul", "progressive hip hop", "country rap", "cybergrind", "chillwave metal",
    
    # Pop & Contemporary
    "bubblegum pop", "alternative R&B", "neo-soul", "dream pop", "indie pop", "synthpop", "hip hop", "R&B", "electropop",
    "indietronica", "alternative dance", "pop-punk", "modern rock", "glam pop", "disco", "dance-pop", "tropical house",
    "synth-pop", "melodic house", "chill-pop", "indie folk", "indie country", "pop-trap", "electro-pop", "pop-folk",
    
    # Experimental & Avant-Garde
    "noise", "free jazz", "avant-garde metal", "glitch", "industrial ambient", "sound collage", "electroacoustic",
    "musique concrète", "prepared piano", "non-music", "microsound", "drone metal", "psychedelic noise", "avant-garde pop",
    "anti-folk", "radical metal", "unclassifiable", "exotica", "serialism", "post-dubstep", "chamber noise", "new media art",
    
    # Other Genres
    "skramz", "space rock", "post-metal", "vaporwave jazz", "anarcho-punk", "dark cabaret", "steamwave", "lo-fi techno",
    "orchestral glitch", "psychedelic trance", "future funk", "gothic rock", "crunk", "anti-folk", "bubblegum bass", 
    "jungle", "witch house", "synthwave", "beatnik jazz", "metalstep", "future soul", "industrial pop", "dark trap", 
    "post-soul", "twerk", "speed metal", "boogie", "contemporary classical", "reggaeton trap", "grime", "swing revival",
    "instrumental hip hop", "shoegaze metal", "old-school rap", "UK grime", "blues-funk", "skatepunk", "post-punk revival",
    "Danish jazz", "futuristic jazz fusion", "minimal wave", "Gothic trance", "electronica", "hyphy", "space ambient", "vocal jazz"
])

music_genres.extend([
    "best bossa nova tracks",            # Bossa nova
    "top funk songs",                    # Funk genre
    "indie rock anthems",                # Indie rock
    "best pop rock songs",               # Pop rock genre
    "synthwave music",                   # Synthwave
    "deep house music",                  # Deep house
    "best world music",                  # World music
    "famous opera arias",                # Opera
    "techno music playlist",             # Techno music
    "best trance music",                 # Trance genre
    "jazz fusion hits",                  # Jazz fusion
    "best folk music",                   # Folk music
    "reggaeton hits",                    # Reggaeton
    "grunge music hits",                 # Grunge rock
    "best rock covers",                  # Rock covers
    "best psychedelic rock",             # Psychedelic rock
    "pop punk anthems",                  # Pop punk
    "best lo-fi hip hop",                # Lo-fi hip hop
    "K-pop hits",                        # K-pop (Korean pop)
    "electro swing music",               # Electro swing
    "progressive rock hits",             # Progressive rock
    "hardcore punk anthems",             # Hardcore punk
    "shoegaze music",                    # Shoegaze genre
    "chiptune music",                    # Chiptune (8-bit music)
    "salsa music",                       # Salsa
    "experimental music",                # Experimental genre
    "triphop music",                     # Trip hop
    "industrial music",                  # Industrial genre
    "gothic rock hits",                  # Gothic rock
    "synthpop classics",                 # Synthpop
    "drum and bass hits",                # Drum and bass
    "downtempo music",                   # Downtempo
    "grime music",                       # Grime (UK hip-hop)
    "electronic body music",             # EBM (Electronic body music)
    "ambient noise music",               # Ambient noise
    "new age music",                     # New age genre
    "raggaeton music",                   # Raggaeton (Latin urban)
    "space rock",                        # Space rock
    "celtic fusion music",               # Celtic fusion
    "zouk music",                        # Zouk (Caribbean music)
    "balkan brass music",                # Balkan brass band music
    "tribal house music",                # Tribal house
    "Tuvan throat singing",              # Tuvan throat singing
    "african drumming music",            # African drumming traditions
    "indian classical music",           # Indian classical
    "arabic electronica",                # Arabic electronic music
    "psytrance music",                   # Psytrance
    "lo-fi beats for studying",          # Lo-fi beats
    "gospel music hits",                 # Gospel
    "blues rock",                        # Blues rock
    "neo-soul music",                    # Neo-soul
    "post-punk music",                   # Post-punk
    "tropical house music",              # Tropical house
    "fusion jazz",                       # Fusion jazz
    "jungle music",                      # Jungle (subgenre of drum and bass)
    "soundtrack music",                  # Film scores and soundtracks
    "ambient trance",                    # Ambient trance
    "tango music",                       # Tango music
    "afro-cuban jazz",                   # Afro-Cuban jazz
    "classic blues standards",           # Classic blues
    "lo-fi chill beats",                 # Lo-fi chill beats
    "flamenco fusion",                   # Flamenco fusion
    "ska music hits",                    # Ska genre
    "avant-garde jazz",                  # Avant-garde jazz
    "experimental rock",                 # Experimental rock

    "Acid Crunk", "Acoustic Doom", "African Futurism", "Ambient Grindcore", "Amoebic Surf", "Angstcore",
    "Apex Jazz", "Arabesque Techno", "Art Noise", "Astral Jazz", "Atmospheric Death Metal", "Avant-Folk",
    "Baroque Metal", "Bebop Hip-Hop", "Biorhythm Music", "Blackgaze", "Bluegrass Fusion", "Bollywood Trap",
    "Brazilian Psytrance", "Cabaret Pop", "Celtic Hardcore", "Chamber Pop", "Chillout Techno", "Chiptune Doom",
    "Chinese Folk Punk", "Cinematic IDM", "Clowncore", "Cloud Rap", "Countrytronica", "Cultural Dub", "Cumbia Jungle",
    "Cyberpunk Jazz", "Dark Ambient House", "Dark Synthwave", "Darkwave Jazz", "Deep Sea Electro", "Desert Blues",
    "Dissonant Techno", "DIY Black Metal", "Drone Doom", "Dubstep Reggae", "Ethereal Hip-Hop", "Folk Noir", "Folk Punk Jazz",
    "Future Funk", "Glitch Jazz", "Gothic Metalcore", "Gothic Pop", "Grindcore Blues", "Hardcore Psytrance",
    "Hippie Techno", "Horrorcore Rap", "Industrial Trap", "Indian Psybient", "Intelligent Techno", "Instrumental Grunge",
    "Intergalactic Jazz", "J-Pop Experimental", "Jazzcore", "Klezmer Metal", "Krautrock Blues", "Lofi House", "Lullaby Metal",
    "Lush Pop", "Macho Disco", "Melodic Black Metal", "Minimal Funk", "Minimal Industrial", "Moldavian Folk", "Monolith Jazz",
    "Mosaic Pop", "Mushroom Jazz", "Neoclassical Darkwave", "Noise Pop", "Outlaw Country Jazz", "Polyrhythmic Metal",
    "Post-Industrial Dub", "Post-Modern Pop", "Post-Punk R&B", "Progressive Ambient", "Progressive Electro", "Psybient Rock",
    "Psychedelic Country", "Psychedelic Soul", "Psychotropic Jazz", "Raga Fusion", "Renaissance Folk", "Retrowave Metal",
    "Riot Grrl Electro", "Russian Jazz", "Shamanic Trance", "Sludgecore Jazz", "Slowcore Psychedelia", "Space Jazz",
    "Spacey Synthpop", "Spiritual Funk", "Synth-Punk", "Synthwave Jazz", "Tango Punk", "Techno-Folk", "Techno-Goth",
    "Thrash Jazz", "Traditional Folk Metal", "Trap Reggae", "Tribal Funk", "Tropical Noise", "Tuvan Metal", "Uplifting Doom",
    "Vaporwave Metal", "Viking Death Metal", "Viking Jazz", "Witch House Pop", "Yodel Trap", "Zeuhl", "Zydeco Funk",
    "Afrobeat Jazz", "Alien Pop", "Balkan Metal", "Balladcore", "Bossa Nova Punk", "Buddhist Rock", "Cabaret Punk",
    "Cinematic Gothic", "Cloud Step", "Country Trap", "Cyber Jazz", "Deep Trap", "Desi Trap", "Dreamwave", "Electric Blues",
    "Electro Swing", "Endless Waves", "Ethno-Jazz", "Experimental Opera", "Folk Doom", "Folk-Gothic", "Future Jazz",
    "Glam Electro", "Gothic Chillwave", "Grunge Jazz", "Hard Acid", "Hard Rock Psychedelic", "Industrial Funk",
    "Japanese Noise", "Jazz-Punk", "K-Pop EDM", "Krautrock Ambient", "Lofi Soul", "Magical Realism Jazz", "Melancholic Metal",
    "Minimal Jazz", "Modern Classical Metal", "Mushroom Techno", "Neon Noir", "New Age Noise", "Nu Jazz Funk",
    "Orchestral Drone", "Outsider Music", "Post-Internet Pop", "Post-Modern Jazz", "Psybient House", "Rave Metal",
    "Retro Electronica", "Romantic Goth", "Scream Jazz", "Shoegaze Punk", "Ska Metal", "Soul Jazz Fusion", "Space-Folk",
    "Spanish Jazz Fusion", "Spiritual Hip-Hop", "Stoner Jazz", "Synth Rock", "Synthwave Blues", "Tango Electro",
    "Tech House Jazz", "Thrash Jazz Fusion", "Tropical Techno", "Urban Jazz", "Vaporwave Classical", "Viking Folk",
    "Visceral Metal", "Zany Jazz", "Zombie Punk", "Zouk Bass", "Acoustic Pop Punk", "Aerobic Disco", "Alpine House",
    "Alpine Punk", "Ambient Breakbeat", "Arabic Trance", "Arctic Dubstep", "Avant-garde Disco", "Balkan Dubstep",
    "Baroque Jazz", "Belly Dance Metal", "Big Beat Blues", "Bossa Nova Chillout", "Brazilian Jazz", "Cinematic Downtempo",
    "Cloud Trap", "Cumbia Hip-Hop", "Dark Psychedelic", "Deep House Trance", "Dirty Funk", "Electro-Pop Jazz", "Ethno Metal",
    "Experimental Flamenco", "Flute Metal", "Folk Blues Rock", "Folk Rap", "Funk Metalcore", "Future Rock", "Glitch Blues",
    "Grindwave", "Hard Techno Jazz", "Industrial Punk", "Indie Noise", "Instrumental Psytrance", "J-Pop Jazz",
    "Jungle Metal", "Kraut Jazz", "Lofi Rap", "Mathcore Blues", "Minimalist Hip-Hop", "Modern Jazz Fusion", "Noise-Trap",
    "Pagan Folk", "Psychedelic Soul Jazz", "Psytrance-Folk", "Rage Rock", "Ragtime Metal", "Retro-Futuristic Jazz",
    "Rocktronica", "Rural Electronic", "Shamanic Ambient", "Ska Reggae", "Sludge Jazz", "Space-Funk", "Spoken Word Jazz",
    "Sunset Dubstep", "Synthwave Punk", "Techno-Dubstep", "Techno-Pop", "Tribal Jazz", "Tribal Trap", "Tropical Chill",
    "Uplifting Psytrance", "Vaporwave Rap", "Viking Electro", "Vintage Punk", "Witchcraft Blues", "Zombie Jazz", "Zeuhl Jazz",
    "Afro-Experimental Jazz", "Artificial Intelligence Music", "Asian Avant-garde", "Bitpop", "Bossa Nova Swing",
    "Celtic Ambient", "Cinematic Trance", "Cyber-Folk", "Darkwave Reggae", "Death Folk", "Deep Space Jazz", "Drone Techno",
    "Dungeon Synth", "Electro-Soul", "Electric Folk", "Folk Jazz", "Fusion Metal", "Gothic Punk", "Instrumental Breakcore",
    "Industrial Trap Fusion", "Mellow Metal", "New Wave Rock", "Nordic Electro", "Post-Metal Jazz", "Progressive Trap",
    "Psytrance Chill", "Rave Jazz", "Rock Opera", "Silent Film Music", "Ska House", "Synth Country", "Techno Ballads",
    "Tribal Electro", "Trippy Hip-Hop", "Underground Trap", "Vaporwave Punk", "Whimsical Jazz", "Witch House Trap",
    "Worldbeat Techno", "Y2K Pop", "Zen Jazz", "Zombie Electronica",

      "Viking Funeral Doom", "Horror Jazz", "Occult Trap", "Melancholic Shoegaze", "Epic Neo-Classical Metal",
    "Alien Jazz Fusion", "Blackened Folk Metal", "Doomstep", "Murder Folk", "Ethereal Post-Metal", "Industrial Appalachian",
    "Hyperborean Black Metal", "Swampy Doom Blues", "Darkwave Flamenco", "Lo-Fi Blackgaze", "Balkan Psychobilly", 
    "Gothic Mathcore", "Baroque Psytrance", "Space Doom Jazz", "Tribal Deathcore", "Deathwave", "Sludgepunk", 
    "Psychedelic Blackgaze", "Witchy Blues", "Hauntology Trap", "Gothic Medieval Techno", "Post-Apocalyptic Jazz", "Hypnotic Death Metal",
    "Dungeon Synthwave", "Folk Noir Jazz", "Retro-Futuristic Jazz Fusion", "Cosmic Doom", "Metallic Dubstep", 
    "Eastern Sludge", "Cinematic Noise Rock", "Folktronica Doom", "Pagan Post-Rock", "Steampunk Electro Swing", "Phantom Pop", 
    "Romantic Sludge", "Noise-Folk", "Neo-Victorian Death Metal", "Atmospheric Deathcore", "Creepy Jazztronica", 
    "Folk Trap", "Lo-Fi Witch House", "Creepy Jazzcore", "Melodic Grindgaze", "Doom-Hop", "Post-Punk Folktronica", 
    "Pre-Raphaelite Doom", "Haunted House Electro", "Psychotic Tango", "Medieval Drone", "Chaotic Folk-Punk", "Bizarro Jazz", 
    "Industrial Psychobilly", "Desert Psybient", "Ritualistic Drone", "Gothic Country", "Cyber-Witchwave", "Romantic Deathcore", 
    "Folk-Pop Noir", "Witch-Pop", "Acoustic Sludge", "Raga Doom Metal", "Transylvanian Jazz", "Gothic R&B", "Glitch-House Ambient",
    "Arabian Doom", "Post-Gothic Jazz", "Hyperreal Ambient", "Post-Punk-Shoegaze Fusion", "Horrorwave", "Folk Doomgaze", 
    "Outsider Jazz", "Dystopian Synthwave", "Tropical Deathcore", "Industrial Country", "Vaporwave Folk", "Gothic Soul", 
    "Ghost Punk", "Suburban Death Metal", "Cinematic Post-Rock", "Lofi Synthwave", "Neon Doom", "Techno-Folk Fusion", 
    "Ambient Folkcore", "Reggae Death Metal", "Southern Gothic Rock", "Cosmic Noise-Pop", "Folk Metalcore", "Cyberspace Doom", 
    "Ritual Pop", "Funk-Punk Doom", "Goth-Pop Shoegaze", "Ethereal Trap Metal", "Grimwave", "Apocalyptic Shoegaze", "Doom Trance", 
    "Futuristic Medieval", "Drone-Folk", "Sludgy Blues-Rock", "Post-Industrial Jazz", "Neo-Experimental Post-Metal", 
    "Electro-Psychedelic Soul", "Future Soul Jazz", "Indie-Country Hybrid", "Folkcore Dubstep", "Hyper-Noise Rock", "Unholy Grunge", 
    "Midnight Wave", "Mystic Trap", "Horrorpunk", "Tectonic Jazz", "Freak Folk", "Ghost Metal", "Melancholic Hardcore", 
    "Celestial Jazzcore", "Space Jazz Fusion", "Weirdcore", "Ambient Doom Jazz", "Dark Neofolk", "Viking Space Metal", 
    "Creepy Lo-Fi Electro", "Abstract Doom", "Gothic Stoner Metal", "Tropical Deathcore", "Cinematic Doomwave", 
    "Sludgy Deathwave", "Pagan Doomcore", "Techno-Folkcore", "Psychotronic Folk", "Noirwave", "Tribal Techno Doom", 
    "Ambient Deathgaze", "Blackened Electro-Rock", "Astro-Pop", "Sacred Doom", "Supernatural Funk", "Hypnotic Techno Metal",
    "Folkwave", "Oceanwave", "Haunting Noisecore", "Neo-Romantic Doom", "Dystopian Gothic Electro", "Futuristic Post-Rock", 
    "Chamber Doom", "Blueswave", "Tribal Darkwave", "Experimental Deathgaze", "Electro-Punk Sludge", "Post-Psychedelic Jazz", 
    "Pagan Psychobilly", "Gothcore",

        "Dubstep", "Brostep", "Riddim", "Chillstep", "Tearout Dubstep", "Deathstep",
    "Future Bass", "Wave", "Melodic Bass", "Chillwave", "Post-Future Bass", "Trap Future Bass",
    "Euphoric Hardstyle", "Rawstyle", "Freeform Hardstyle", "Jumpstyle", "Hard Trance", "Hardcore Hardstyle",
    "Drum & Bass", "Liquid Drum & Bass", "Neurofunk", "Jungle", "Jump Up", "Techstep",
    "Progressive House", "Melodic Progressive House", "Big Room Progressive House", "Deep Progressive House", "Tribal Progressive House", "Progressive Trance",
    "Techno", "Minimal Techno", "Acid Techno", "Detroit Techno", "Industrial Techno", "Hard Techno",
    "Deep House", "Tropical House", "Tech House", "Soulful House", "Progressive House", "Chicago House",
    "Electro House", "Dutch House", "Big Room House", "Funky Electro House", "Electro Swing", "Progressive Electro",
    "Tropical House", "Deep Tropical House", "Future Tropical House", "Dancehall Tropical", "Chill Tropical House", "Afro-Tropical House",

        # Film Music / Epic Orchestral / Video Game Music
    'Epic Orchestral',
    'Cinematic Orchestra',
    'Trailer Music',
    'Epic Hybrid Orchestral',
    'Film Noir',
    'Murder Mystery Score',
    'Historical Film Scores',
    'Fantasy Film Score',
    'Romantic Film Score',
    'Orchestral VGM',
    'Chiptune / 8-Bit Music',
    'Ambient VGM',
    'Electro / Synthwave VGM',
    'Action Adventure VGM',
    'Cinematic VGM',
    'Symphonic Metal',
    'Battle Hymns / War Music',
    'Space Opera Soundtrack',
    'Choral and Gregorian Orchestral',
    'Neoclassical Cinematic',
    'Gothic Orchestral',
    'Industrial Cinematic',
    'Ambient Minimalism',
    'Piano-Driven Film Score',
    'Avant-Garde Cinematic',
    'Modular Synth VGM',
    'World Music Orchestral',
    'Jazz-Fusion Film Score',
    'Dark Electronic Soundtrack',
    'Futuristic Soundscapes',
    'Post-Rock / Cinematic Hybrid',
    'Folk Orchestral',
    
    # Psychedelic and Experimental Genres
    'Dark Psychedelic',
    'Post-Psychedelic Jazz',
    'Psychedelic Soul',
    'Psychedelic Trance',
    'Vintage Psychedelic',
    'Neo-Psychedelia',
    'Psychedelic Rock',
    'Space Rock',
    'Drone Metal',
    'Stoner Rock',
    'Experimental Rock',
    'Noise Rock',
    'Post-Punk / Psychedelic Punk',
    'Industrial Noise',
    'Glitch Art',
    
    # World Music and Folk Subgenres
    'World Fusion',
    'Afrobeat',
    'Bossa Nova',
    'Latin Jazz',
    'Flamenco Fusion',
    'Celtic Folk',
    'Arabic Fusion',
    'Indian Classical Fusion',
    'Tropicalia',
    'Fado',
    
    # Jazz Subgenres
    'Free Jazz',
    'Bebop',
    'Jazz Fusion',
    'Smooth Jazz',
    'Jazz Funk',
    'Hard Bop',
    'Gypsy Jazz',
    'Latin Jazz',
    'Big Band Swing',
    
    # Electronic Subgenres
    'Ambient Techno',
    'Minimal Techno',
    'Acid House',
    'Future Bass',
    'Dubstep',
    'Trap Music',
    'Progressive Trance',
    'Hardstyle',
    'Vaporwave',
    'Chillwave',
    'Synthwave',
    'Lo-Fi Hip Hop',
    
    # Metal Subgenres
    'Symphonic Metal',
    'Black Metal',
    'Death Metal',
    'Doom Metal',
    'Sludge Metal',
    'Power Metal',
    'Thrash Metal',
    'Gothic Metal',
    'Folk Metal',
    'Avant-garde Metal',
    
    # Rock Subgenres
    'Indie Rock',
    'Post-Rock',
    'Garage Rock',
    'Psychedelic Rock',
    'Hard Rock',
    'Alternative Rock',
    'Progressive Rock',
    'Post-Hardcore',
    'Math Rock',
    'Emo',
    
    # Classical and Neoclassical Subgenres
    'Baroque',
    'Classical Symphony',
    'Opera',
    'Romantic Classical',
    'Neoclassical',
    'Minimalist Classical',
    'Modern Classical',
    'Contemporary Classical',
    
    # Hip Hop Subgenres
    'Boom Bap',
    'Trap',
    'Gangsta Rap',
    'Conscious Rap',
    'Lo-Fi Hip Hop',
    'Alternative Hip Hop',
    'West Coast Hip Hop',
    'East Coast Hip Hop',
    'Cloud Rap',
    'Mumble Rap',
    
    # Blues Subgenres
    'Delta Blues',
    'Chicago Blues',
    'Electric Blues',
    'Jump Blues',
    'Country Blues',
    'Texas Blues',
    
    # Country Subgenres
    'Alt-Country',
    'Bluegrass',
    'Honky Tonk',
    'Outlaw Country',
    'Country Rock',
    'Americana',
    
    # Dance Subgenres
    'House',
    'Progressive House',
    'Deep House',
    'Tech House',
    'Electro House',
    'Trance',
    'New Hardstyle',
    'Techno',
    'Dubstep',
    
    # Other Unique Genres
    'New Age',
    'Chamber Pop',
    'Gothic Rock',
    'Post-Industrial',
    'Soundtrack Rock',
    'Industrial Rock',
    'Ska',
    'Reggae Fusion',
    'Salsa',
    'Cumbia',
    'K-Pop',
    'J-Pop',
    'C-Pop',
    'Tuvan Throat Singing',
    'Experimental Jazz',
    'Jazz Noir',
    'Indie Folk',
    'Jazz-Rock Fusion',
    'Psych Folk',

    'Noise Music',
    'Musique Concrète',
    'Microtonal Music',
    'Glitch Music',
    'Biofeedback Music',
    'Circuit Bending',
    'Chamber Music for Electronics',
    'Tuvan Throat Singing',
    'Algorave',
    'Vaporwave Revival',
    'Futurist Sound Art',
    'Phantom Soundscapes',
    'Plunderphonics',
    'Acousmatic Music',
    'Fourth World Music',
    'Ambient Drone Metal',
    'Psychogeographical Soundscapes',
    'Transcendental Sound Therapy',
    'Hyperpop',
    'Jazztronica',
    'Low-Fi Sound Design',
    'Deep Listening',
    'Medieval / Renaissance Electronic Fusion',
    'Kirlian Photography Soundtrack',
    'Dark Ambient Black Metal Fusion',
    'Cybergrind',
    'Quirky Soundtrack Pop',
    'Haptic Music',
    'Nature Sound Studies',
    'Time-Stretched Soundscapes',

       'Noise Music', 
    'Musique Concrète', 
    'Microtonal Music', 
    'Glitch Music', 
    'Biofeedback Music', 
    'Circuit Bending', 
    'Chamber Music for Electronics', 
    'Tuvan Throat Singing', 
    'Algorave', 
    'Vaporwave Revival', 
    'Futurist Sound Art', 
    'Phantom Soundscapes', 
    'Plunderphonics', 
    'Acousmatic Music', 
    'Fourth World Music', 
    'Ambient Drone Metal', 
    'Psychogeographical Soundscapes', 
    'Transcendental Sound Therapy', 
    'Hyperpop', 
    'Jazztronica', 
    'Low-Fi Sound Design', 
    'Deep Listening', 
    'Medieval / Renaissance Electronic Fusion', 
    'Kirlian Photography Soundtrack', 
    'Dark Ambient Black Metal Fusion', 
    'Cybergrind', 
    'Quirky Soundtrack Pop', 
    'Haptic Music', 
    'Nature Sound Studies', 
    'Time-Stretched Soundscapes', 
    
    # Video Game Music and Related Genres
    'Chiptune', 
    '8-bit Music', 
    '16-bit Music', 
    'Video Game Soundtrack', 
    'VGM Remixes', 
    'Lo-fi Video Game Music', 
    'Dungeon Synth', 
    'Symphonic Video Game Music', 
    'Video Game Jazz', 
    'Orchestral Score Music', 
    
    # Horror and Dark Genres
    'Dark Ambient', 
    'Industrial Horror', 
    'Horror Soundtracks', 
    'Surreal Horror Music', 
    'Death Industrial', 
    'Creepy Soundscapes', 
    'Psychological Horror Soundtracks', 
    'Gothic Horror Music', 
    
    # Film and Instrumental Music Genres
    'Cinematic Orchestral', 
    'Epic Orchestral', 
    'Film Score', 
    'Post-Rock Film Soundtracks', 
    'Experimental Film Music', 
    'Instrumental Post-Rock', 
    'Minimalist Film Soundtrack', 
    'Avant-Garde Classical', 
    'Neo-Classical Soundtracks', 
    'Instrumental Ambient',

      # Africa
    "Traditional African Music",
    "Highlife (Ghana)",
    "Mbira (Zimbabwe)",
    "Gnawa (Morocco)",
    "Zulu (South Africa)",
    "Ewe (Ghana/Togo)",
    "Mali Blues (Mali)",
    "Afrobeat (Nigeria)",
    "Benga (Kenya)",
    "Soukous (Democratic Republic of Congo)",
    "Sabar (Senegal)",
    "Mbalax (Senegal)",
    "Chimurenga (Zimbabwe)",
    "Amapiano (South Africa)",
    "Kwaito (South Africa)",
    "Kora Music (West Africa)",
    "Bajau Music (Philippines/Indonesia)",

    # Latin America
    "Cumbia (Colombia)",
    "Tango (Argentina)",
    "Mariachi (Mexico)",
    "Salsa (Cuba/Latin America)",
    "Reggaeton (Puerto Rico)",
    "Bossa Nova (Brazil)",
    "Forró (Brazil)",
    "Mambo (Cuba)",
    "Samba (Brazil)",
    "Ranchera (Mexico)",
    "Choro (Brazil)",
    "Vallenato (Colombia)",
    "Banda (Mexico)",
    "Corrido (Mexico)",
    "Cajón Music (Peru)",
    "Andean Music (Peru/Bolivia/Chile)",

    # Caribbean and Central America
    "Merengue (Dominican Republic)",
    "Danzón (Cuba)",
    "Calypso (Caribbean)",
    "Reggae (Jamaica)",
    "Soca (Trinidad & Tobago)",
    "Zouk (Caribbean)",
    "Kaiso (Caribbean)",
    
    # Indigenous Music
    "Native American Music (USA)",
    "Australian Aboriginal Music",
    "Siberian/Tuvan Throat Singing (Siberia)",
    "Ainu Music (Japan)",
    "Andean Music (Peru/Bolivia/Chile)",
    "Hmong Music (Southeast Asia)",
    "Māori Music (New Zealand)",
    "Navajo Music (USA)",
    "Inuit Throat Singing (Arctic regions)",
    "Lakota Music (USA)",
    
    # Europe
    "Celtic (Irish/Scottish) Folk",
    "Flamenco (Spain)",
    "Klezmer (Eastern Europe)",
    "Nordic Folk (Scandinavia/Iceland)",
    "Scandinavian Folk (Norway/Sweden)",
    "Icelandic Folk",
    "Faroese Folk (Faroe Islands)",
    "Balkan Folk (Balkans)",
    "Gothic Folk (Germany/UK)",
    "Polka (Central/Eastern Europe)",
    "Cajun Music (Louisiana, USA)",
    "Madrigal (Italy/Spain)",
    "Sevdalinka (Balkans)",
    "Lautari Music (Romania)",
    "Dombra Music (Kazakhstan)",

    # Middle East & Asia
    "Maqam (Arab World)",
    "Raga (India)",
    "Gamelan (Indonesia)",
    "Qawwali (Pakistan/India)",
    "Khyal (India)",
    "Fado (Portugal)",
    "Dastgah (Iran)",
    "Bhangra (India/Pakistan)",
    "Taqsim (Middle East)",
    "Raï (Algeria)",
    "Sufi Music (Middle East)",
    "Caspian Music (Iran/Azerbaijan)",
    "Chetan (China)",
    "Sitar Music (India)",
    "Qanun (Middle East)",
    "Dholak Music (India)",
    
    # East Asia & Southeast Asia
    "K-Pop (South Korea)",
    "J-Pop (Japan)",
    "Mandopop (Taiwan)",
    "C-Pop (China)",
    "Taoist Chanting (China)",
    "Shamisen Music (Japan)",
    "Gagaku (Japan)",
    "Chinese Folk Music (China)",
    "Tuvan Throat Singing (Mongolia)",
    "Pinoy Music (Philippines)",
    "Guitar Cumbia (Mexico/Philippines)",
    "Indo-Jazz (India)",
    "Thai Luk Thung (Thailand)",
    "Batuque (Angola)",
    "Vietnamese Traditional Music (Vietnam)",

    # Microgenres / Internet-born Genres
    "Vaporwave",
    "Chillwave",
    "Lo-Fi Hip-Hop",
    "Cloud Rap",
    "Hyperpop",
    "SoundCloud Rap",
    "Synthwave",
    "Future Funk",
    "Darkwave",
    "Industrial Music (Cyberpunk)",
    "Trap (Atlanta/USA)",
    "Post-Internet Pop",
    "Skramz (Emo)",
    "Drill (UK/Chicago)",
    "Bedroom Pop",
    "Lo-Fi Indie",

    # seasonal music
    "Christmas Music",
    "Holiday Music",
    "Winter Music",
    "Hanukkah Music",
    "Festive Music",
    "New Year's Eve Music",
    "Easter Music",
    "Spring Music",
    "Summer Music",
    "Autumn Music",
    "Halloween Music",
    "Valentine's Day Music",
    "Thanksgiving Music",
    "Diwali Music",
    "Ramadan Music",
    "Solstice Music",
    "Seasonal Folk Music",
    "Winter Solstice Music",
    "Beach Music",
    "Summer Pop",
    "Spring Jazz",
    "Fall Blues",
    "Autumn Indie",
    "Christmas Jazz",
    "Christmas Rock",
    "Christmas Carols",
    "Christmas Classical",
    "Winter Acoustic",
    "Winter Jazz",
    "Ski Resort Music",
    "Autumn Chillwave",
    "Halloween Horror Music",
    "Autumn Acoustic",
    
    # Folk and Traditional Music Expansion
    "Celtic (Irish/Scottish) Folk",
    "Flamenco (Spain)",
    "Klezmer (Eastern Europe)",
    "Nordic Folk (Scandinavia/Iceland)",
    "Scandinavian Folk (Norway/Sweden)",
    "Icelandic Folk",
    "Faroese Folk (Faroe Islands)",
    "Appalachian Folk (USA)",
    "Tuvan Throat Singing (Mongolia/Siberia)",
    "Balkan Folk (Balkans)",
    "Ghazal (South Asia)",
    "Maqam (Arab World)",
    "Raga (India)",
    "Gamelan (Indonesia)",
    "Tango Nuevo (Argentina)",
    "Polynesian Traditional Music",
    "Malagasy Music (Madagascar)",
    "Bajau Traditional Music (Philippines/Indonesia)",
    "Saami Joik (Northern Europe)",
    "Māori Haka (New Zealand)",
    "Fijian Traditional Music (Fiji)",

    "hardstyle top 10 best",
    
    "Indian classical music", "Korean court music", "Persian classical music", 
    "Ottoman music", "Western classical music", "Early music", "Medieval music", 
    "Ars antiqua", "Ars nova", "Ars subtilior", "Renaissance music", "Baroque music", 
    "Galant music", "Classical period", "Romantic music", "20th and 21st-centuries classical music", 
    "Modernism", "Impressionism", "Neoclassicism", "High modernism", "Postmodern music", 
    "Experimental music", "Contemporary classical music", "Minimal music", 
    "Popular", "Avant-garde & experimental", "Crossover music", "Danger music", 
    "Drone music", "Electroacoustic", "Industrial music", "Instrumental", "Lo-fi", 
    "Musical improvisation", "Musique concrète", "Noise", "Outsider music", "Progressive music", 
    "Psychedelic music", "Underground music", "Blues", "African blues", "Blues rock", 
    "British blues", "Canadian blues", "Chicago blues", "Classic female blues", "Contemporary R&B", 
    "Country blues", "Delta blues", "Desert blues", "Detroit blues", "Electric blues", 
    "Gospel blues", "Hill country blues", "Hokum blues", "Jump blues", "Kansas City blues", 
    "Louisiana blues", "Memphis blues", "New Orleans blues", "Piedmont blues", "Punk blues", 
    "Rhythm and blues", "Doo-wop", "Soul blues", "St. Louis blues", "Swamp blues", "Talking blues", 
    "Texas blues", "West Coast blues", "Country", "Alternative country", "Americana", 
    "Cowpunk/Country-punk", "Gothic country", "Roots rock", "Australian country", "Bush band", 
    "Bakersfield sound", "Bluegrass", "Old-time bluegrass", "Traditional bluegrass", 
    "Progressive bluegrass", "Bluegrass gospel", "Blue Yodeling", "Bro-country", "Cajun", 
    "Cajun fiddle", "Canadian country", "Franco-country", "Christian country", "Classic country", 
    "Country and Irish", "Country blues", "Country en Español", "Country folk", "Country pop", 
    "Country rap", "Country rock", "Cowboy pop", "Dansband", "Gulf and Western", "Hokum", 
    "Honky tonk", "Instrumental country", "Lubbock sound", "Nashville sound", "Countrypolitan", 
    "Neotraditional country", "New country", "Old-time", "Outlaw country", "Pop country", 
    "Progressive country", "Regional Mexican", "Rockabilly", "Psychobilly", "Gothabilly", 
    "Southern rock", "Southern soul", "Sertanejo", "Talking blues", "Traditional country", 
    "Truck-driving country", "Western music", "New Mexico music", "Red dirt", "Tejano", "Texas country", 
    "Western swing", "Zydeco", "Easy listening", "Adult contemporary music", "Adult standards", 
    "Background music", "Elevator music", "Barococo", "Beautiful music", "Chill-out", "Downtempo", 
    "Furniture music", "Light music", "Lounge music", "Middle of the road", "New-age music", 
    "Soft rock", "Electronic", "Ambient", "Ambient dub", "Dark ambient", "Ambient industrial", 
    "Dungeon synth", "Isolationism", "Dreampunk", "Illbient", "New-age", "Neoclassical new-age", 
    "Space music", "Bass music", "Footwork", "Future bass", "Kawaii future bass", "Jungle terror", 
    "Midtempo bass", "Trap", "UK bass", "Wave", "Hardwave", "Breakbeat", "Acid breaks", "Baltimore club", 
    "Jersey club", "Philly club", "Big beat", "Breakbeat hardcore", "Darkcore", "Hardcore breaks", 
    "Broken beat", "Florida breaks", "Nu skool breaks", "Progressive breaks", "Psychedelic breakbeat", 
    "Chill-out", "Downtempo", "Psybient", "Psydub", "Trip hop", "Trip rock", "Disco", "Afro/cosmic music", 
    "Electro-disco", "Hi-NRG", "Eurobeat", "Eurodance", "Italo dance", "Italo disco", "Spacesynth", 
    "Space disco", "Eurodisco", "Nu-disco", "Post-disco", "Boogie", "City pop", "Drum and bass", 
    "Darkstep", "Drumfunk", "Drumstep", "Hardstep", "Intelligent drum and bass", "Atmospheric drum and bass", 
    "Jazzstep", "Jump-up", "Liquid funk", "Neurofunk", "Sambass", "Techstep", "Dub", "Dub poetry", 
    "Dubtronica", "Electronic rock", "Dance-rock", "Alternative dance", "Baggy", "New rave", "Dance-punk", 
    "Electronic pop", "Dance-pop", "Freestyle", "Disco polo", "Hyperpop", "Sophisti-pop", "Synth-pop", 
    "Electroclash", "Electropop", "Wonky pop", "Indietronica", "Krautrock", "New wave", "Cold wave", 
    "Dark wave", "Neoclassical dark wave", "Neue Deutsche Todeskunst", "Ethereal wave", "Nu-gaze", 
    "Minimal wave", "Neue Deutsche Welle", "New romantic", "Post-rock", "Space rock", "Synth-metal", 
    "Electrogrind", "Electronicore", "Synth-punk", "Electronica", "Folktronica", "Live electronic", 
    "Laptronica", "Nu jazz", "Progressive electronic", "Berlin School", "Kosmische musik", 
    "Ethnic electronica", "Asian Underground", "African electronic dance music", "Afrobeats", "Azonto", 
    "Coupé-décalé", "Kuduro", "Mahraganat", "Shangaan electro", "Budots", "Changa tuki", "Dancehall pop", 
    "Denpa music", "Guaracha", "Funk carioca", "Funk melody", "Funk ostentação", "Proibidão", "Rasteirinha", 
    "Merenhouse", "Nortec", "Rabòday", "Rara tech", "Russ music", "Shamstep", "Tecnocumbia", 
    "Tribal guarachero", "Worldbeat", "Manila sound", "Experimental electronic", "Black MIDI", 
    "Deconstructed club", "Drone", "Electroacoustic music", "Acousmatic music", "Electroacoustic improvisation", 
    "Musique concrète", "Soundscape", "Glitch", "Microsound", "Noise music", "Danger music", "Japanoise", 
    "Harsh noise", "Harsh noise wall", "Power electronics", "Death industrial", "Power noise", "Plunderphonics", 
    "Sampledelia", "Reductionism", "Lowercase", "Onkyokei", "Funk fusion", "Acid jazz", "Funktronica", 
    "Synth-funk", "Jungle", "Ragga jungle", "Hard dance", "Hard NRG", "Hardstyle", "Dubstyle", 
    "Euphoric frenchcore", "Euphoric hardstyle", "Rawstyle", "Trapstyle", "Jumpstyle", "Lento violento", 
    "Mákina", "Hardcore", "Bouncy techno", "Breakcore", "Raggacore", "Digital hardcore", "Frenchcore", 
    "Gabber", "Early hardcore", "Mainstream hardcore", "Happy hardcore", "UK hardcore", "Industrial hardcore", 
    "J-core", "Speedcore", "Extratone", "Flashcore", "Splittercore", "Hauntology", "Chillwave", 
    "Hypnagogic pop", "Synthwave", "Darksynth", "Sovietwave", "Vaporwave", "Future funk", "Hardvapour", 
    "Mallsoft", "Hip hop fusion", "Afroswing", "Alternative hip hop", "Hipster hop", "Cloud rap", "Crunk", 
    "Crunkcore", "Snap music", "Electro", "Emo rap", "Glitch hop", "Instrumental hip hop", "Lofi hip hop", 
    "Miami bass", "Mumble rap", "Trap", "Afro trap", "Drill", "Brooklyn drill", "UK drill", "Latin trap", 
    "Phonk", "Drift phonk", "Brazilian phonk", "Plugg", "UK trap", "House music", "Acid house", "Afro house", 
    "Afro tech", "Amapiano", "Kidandali", "Deep house", "Detroit techno", "French house", "Hard house", 
    "Hard trance", "House trance", "Italo house", "Miami bass", "Progressive house", "Soulful house", 
    "Tropical house", "Trance music", "Hard trance", "Eurotrance", "Progressive trance", "Psytrance", 
    "Techno", "Detroit techno", "Minimal techno", "Hard techno", "Industrial techno", "Progressive techno", 
    "Tech-house", "Deep techno", "Dub techno", "Tribal techno", "Garage techno", "Dubstep", "Brostep", 
    "Post-dubstep", "Grime", "Future garage", "Heavy dubstep", "Skweee", "Wobble bass", "Kuduro", 
    "Post-dubstep", "Swingbeat", "Brokenbeat", "Jazztronica", "Ambient techno", "UK garage", "Deep house", 
    "Tropical house", "Tech house", "Electronica", "Indie rock", "Alternative rock", "Hard rock", "Post-punk", 
    "Punk rock", "Garage rock", "Indie pop", "Dream pop", "Shoegaze", "Pop rock", "Folk rock", "Folk punk", 
    "Psychedelic rock", "Hardcore punk", "Math rock", "Emo", "Noise rock", "Post-hardcore", "Grunge", 
    "Sludge metal", "Stoner rock", "Doom metal", "Black metal", "Death metal", "Deathcore", "Metalcore", 
    "Power metal", "Thrash metal", "Speed metal", "Progressive metal", "Symphonic metal", "Folk metal", 
    "Gothic metal", "Industrial metal", "Nu metal", "Alternative metal", "Sludge metal", "Hardcore punk", 
    "Grindcore", "Blackgaze", "Post-black metal", "Noise rock", "Jazz rock", "Blues rock", "Country rock", 
    "New wave", "Post-punk", "Garage rock", "Glam rock", "Psychedelic rock", "Synth rock", "Progressive rock", 
    "Indie pop", "Dream pop", "Shoegaze", "Trip hop", "New wave", "Pop punk", "Melodic hardcore", 
    "Hardcore punk", "Punk rock"

     "Video game music",
    "Chiptune",
    "Bitpop",
    "Skweee",
    "Nintendocore",
    "FM synthesis",
    "Sequencer music",
    "Hard dance",
    "Hardstyle",
    "Dubstyle",
    "Frenchcore",
    "Trapstyle",
    "Jumpstyle",
    "Lento violento",
    "Mákina",
    "Experimental electronic",
    "Black MIDI",
    "Deconstructed club",
    "Drone",
    "Electroacoustic music",
    "Acousmatic music",
    "Musique concrète",
    "Soundscape",
    "Industrial hip hop",
    "Cyber metal",
    "Neue Deutsche Härte",
    "Witch house",
    "Intelligent dance music (IDM)",
    "Algorave",
    "Drill 'n' bass",
    "Afroswing",
    "Alternative hip hop",
    "Cloud rap",
    "Crunk",
    "Snap music",
    "Electro",
    "Emo rap",
    "Glitch hop",
    "Instrumental hip hop",
    "Lofi hip hop",
    "Miami bass",
    "Mumble rap",
    "Trap",
    "Afro trap",
    "Drill",
    "Brooklyn drill",
    "UK drill",
    "Latin trap",
    "Afrobeat",
    "Reggaeton",
    "Salsa",
    "Cumbia",
    "Calypso",
    "Tropical House",
    "Baile Funk",
    "Dancehall",
    "Disco",
    "Boogie",
    "Hi-NRG",
    "Eurobeat",
    "Eurodance",
    "Italo dance",
    "Italo disco",
    "Spacesynth",
    "Space disco",
    "Nu-disco",
    "Post-disco",
    "Funk",
    "Jazz Funk",
    "Soul",
    "Blues",
    "R&B",
    "Neo-Soul",
    "Psychedelic Soul",
    "Electro Swing",
    "Bossa Nova",
    "Smooth Jazz",
    "Swing",
    "Big Band",
    "Gospel",
    "Jazz",
    "Avant-garde Jazz",
    "Ethio Jazz",
    "Fusion Jazz",
    "Progressive Soul",
    "Folk",
    "Indie Folk",
    "Folk Rock",
    "Blues Rock",
    "Southern Rock",
    "Psych-folk",
    "World Music",
    "Afro/cosmic music",
    "Reggae",
    "Rocksteady",
    "Ska",
    "Post-punk",
    "Art Punk",
    "Post-Rock",
    "Shoegaze",
    "Dream Pop",
    "Indie Pop",
    "Math Rock",
    "Grunge",
    "Hardcore",
    "Noise Rock",
    "Garage Rock",
    "Pop",
    "Synthwave",
    "Retrowave",
    "Chillwave",
    "Deep House",
    "Tech House",
    "Progressive House",
    "Future Bass",
    "House",
    "Techno",
    "Minimal Techno",
    "Detroit Techno",
    "Berlin Techno",
    "Ambient Techno",
    "Acid House",
    "Hardcore Techno",
    "Trance",
    "Psytrance",
    "Goa Trance",
    "Uplifting Trance",
    "Progressive Trance",
    "Drum and Bass",
    "Jungle",
    "Dubstep",
    "Chillout",
    "New Age",
    "Post-Rock",
    "Space Rock",
    "Indie Rock",
    "Power Pop",
    "Noise Pop",
    "Mathcore",
    "Djent",
    "Post-Metal",
    "Sludge",
    "Deathcore",
    "Grindcore",
    "Speed Metal",
    "Thrash Metal",
    "Progressive Metal",
    "Symphonic Metal",
    "Gothic Metal",
    "Industrial Rock",
    "Cybergrind",
    "Noise Metal",
    "Psychedelic Rock",
    "Stoner Rock",
    "Sludge Metal",
    "Alt-Country",
    "Country Rock",
    "Americana",
    "Bluegrass",
    "Progressive Country",
    "Pop Punk",
    "Emo",
    "Hardcore Punk",
    "Folk Punk",
    "Power Pop",
    "Melodic Hardcore",
    "Ska Punk",
    "Post Hardcore",
    "Math Rock",
    "Tribal House",
    "Jazz House",
    "Electro Funk",
    "Tech Funk",
    "Freestyle",
    "Future Funk",
    "Vaporwave",
    "Gothic Rock",
    "Gothic Wave",
    "Coldwave",
    "Minimal Wave",
    "Post-punk Revival",
    "Ambient Pop",
    "Indie Electronica",
    "Tropical Funk",
    "Nu Jazz",
    "Dub",
    "Dub Reggae",
    "Psychill",
    "Ambient House",
    "Indietronica",
    "Folk Metal",
    "Trip Hop",
    "Space Funk",
    "Trap Soul",
    "Tropical Funk",
    "Ethno House",
    "Experimental Techno",
    "Indie Dance",
    "Disco House",
    "Celtic Folk",
    "Urban Pop",
    "Hyperpop",
    "Alt-Pop",
    "Futuristic Funk",
    "Future Garage",
    "Trap EDM",

    
    # Violin and related sub-genres
    "Violin",
    "Classical Violin",
    "Baroque Violin",
    "Romantic Violin",
    "Contemporary Violin",
    "Violin Concerto",
    "Violin Sonatas",
    "Folk Violin",
    "Celtic Violin",
    "Bluegrass Violin",
    "Jazz Violin",
    "Gypsy Violin",
    "Electric Violin",
    "Violin + Electronic Music",
    "Neo-Classical Violin",
    "Progressive Violin",
    "Violin in Film Scores",
    "Violin in Video Game Music",
    "Fusion Violin",
    "Violin Ambient",
    "Experimental Violin",
    "Avant-garde Violin",
    "Violin Looping",
    "Punk Violin",
    "Indie Violin",

    # Trumpet and related sub-genres
    "Trumpet",
    "Classical Trumpet",
    "Jazz Trumpet",
    "Latin Jazz Trumpet",
    "Orchestral Trumpet",
    "Trumpet Solo",
    "Brass Quintet",
    "Marching Band Trumpet",
    "Funk Trumpet",
    "Big Band Trumpet",
    "Trumpet in Film Scores",
    "Trumpet + Electronic Music",
    "New Orleans Trumpet",
    "Trumpet with Improvisation",
    "Trumpet in Soul Music",
    "Trumpet in Blues",

    # UK Hip Hop, Trap, Drill and related sub-genres
    "UK Hip Hop",
    "Grime",
    "UK Trap",
    "Drill",
    "Mumble Rap UK",
    "UK Drill Bounce",
    "Trap Metal UK",
    "UK Dubstep Trap",
    "Trap Drill Fusion",
    "UK Hip Hop Lo-fi",

    # Funk and related sub-genres
    "Funk",
    "Classic Funk",
    "P-Funk",
    "Soul Funk",
    "Funk Rock",
    "Funk Metal",
    "Funk R&B",
    "Funk Instrumental",
    "Brazilian Funk",
    "Indie Funk",

    # Reggae and related sub-genres
    "Reggae",
    "Roots Reggae",
    "Dub",
    "Dancehall",
    "Ska",
    "Rocksteady",
    "Reggaeton",
    "Jungle Reggae",
    "Reggae Blues",
    "World Reggae",
    "Ska Punk",
    "Dub Poetry",
    "Roots Dub",
    "Psychadelic Reggae",

    # Ambient and related sub-genres
    "Ambient",
    "Dark Ambient",
    "Drone Ambient",
    "Ambient Electronica",
    "Ambient Noise",
    "Trance Ambient",
    "Deep Ambient",
    "Synthwave Ambient",
    "Post-Rock Ambient",
    "Ambient Glitch",

    # Afrobeat and related sub-genres
    "Afrobeat",
    "Afrobeat Hip Hop",
    "Afrobeat Electronica",
    "Afrobeat Electronic Funk",
    "Afrobeat Psychedelic",
    "Afrobeat R&B",
    "Afrobeat World",
    "Afrobeat Indie",
    "Afrobeat Reggaeton",
    "Afrobeat Ambient",

        # Video Game Soundtracks
    "Orchestral Fantasy Adventure (The Elder Scrolls V: Skyrim)",
    "Synthwave with Orchestral Elements (Cyberpunk 2077)",
    "Ambient Atmospheric (Journey)",
    "Chiptune Synth-Pop (Hotline Miami)",
    "Cinematic Orchestral with Epic Choirs (The Last of Us Part II)",
    "Jazz and Swing Fusion (Bioshock Infinite)",
    "Electronic Industrial with Dark Ambiance (DOOM 2016)",
    "Neo-Classical Piano with Minimalist Touches (Nier: Automata)",
    "Experimental Sound Design with Ambient (Outer Worlds)",
    "Epic Orchestral with Choir (The Witcher 3: Wild Hunt)",
    "Retro Chiptune with Emotional Depth (Celeste)",
    "Orchestral Adventure with Medieval Influence (Dark Souls)",
    "Ambient Drone Soundscapes (The Vanishing of Ethan Carter)",
    "Folk-Oriented Soundtrack (Red Dead Redemption 2)",
    "Progressive Electronic Rock (Transistor)",
    "Cinematic Orchestral with Rock Elements (Final Fantasy XV)",
    "Acoustic Guitar and Piano Blend (Death Stranding)",
    "Baroque Classical and Electronic Hybrid (The Talos Principle)",
    "Futuristic Cyberpunk Electro (Deus Ex: Human Revolution)",
    "Dynamic Percussion-Driven Soundtrack (God of War 2018)",
    
    # Film Soundtracks
    "Epic Orchestral with Dramatic Choirs (Gladiator)",
    "Ambient Atmospheric with Electronica (Blade Runner 2049)",
    "Minimalist Piano-Driven Drama (The Pianist)",
    "Futuristic Synth-Pop and Electro (Drive)",
    "Romantic Orchestral Ballads (The Shape of Water)",
    "Dark Atmospheric Neo-Noir (Se7en)",
    "Orchestral Action and Drama (Inception)",
    "Cinematic Rock and Jazz Fusion (Guardians of the Galaxy)",
    "Chamber Orchestra with Choral Elements (The Lord of the Rings Trilogy)",
    "Epic Orchestral Action with Strong Percussion (The Dark Knight)",
    "Indie Folk Acoustic with Piano (Little Miss Sunshine)",
    "Psychological Thriller with Dark Ambience (Requiem for a Dream)",
    "Baroque and Classical Fusion (The Favourite)",
    "Ambient Electro-Folk Soundtrack (Her)",
    "Ambient Synth-Driven Horror (Annihilation)",
    "Funk and Soul Film Score (The Big Lebowski)",
    "Industrial Gothic with Orchestral Elements (The Crow)",
    "Jazz-Infused Crime Thriller (L.A. Noire)",
    "Experimental Electro-Orchestral (Under the Skin)",
    "Synthwave Soundtrack with Retro Elements (Stranger Things)",
    "Gothic Orchestral with Choir (Crimson Peak)",
    
    # TV Show Soundtracks
    "Cinematic and Nostalgic Synth (Stranger Things)",
    "Ambient Dark and Atmospheric (True Detective, Season 1)",
    "Psychedelic Rock and Jazz Fusion (Twin Peaks)",
    "Folk-Inspired and Soulful (Big Little Lies)",
    "Tension-Building Electronic with Minimalist Elements (Breaking Bad)",
    "Orchestral with Tribal Influence (Game of Thrones)",
    "Classic Detective Jazz with Blues (Sherlock)",
    "80s Pop and Electro (The Americans)",
    "Neo-Classical Orchestral Drama (Chernobyl)",
    "Acoustic Folk and Minimalist (The Crown)",
    "Industrial Ambient for Sci-Fi Thriller (Westworld)",
    
    # Anime Soundtracks
    "Synthwave with Retro 80s Vibes (Cowboy Bebop)",
    "Epic Orchestral with Japanese Traditional Elements (Spirited Away)",
    "Cinematic Post-Rock with Haunting Melodies (Attack on Titan)",
    "Synth-Pop and Jazz Fusion (Neon Genesis Evangelion)",
    "Chiptune with Experimental Sound Design (FLCL)",
    "Orchestral Fantasy with Eastern Influence (Princess Mononoke)",
    "Orchestral Rock and Punk Influences (Samurai Champloo)",
    "Minimalist Piano with Atmospheric Ambience (Your Name)",
    "Dreamlike Ambient Soundscapes (Made in Abyss)",
    "Futuristic Electronic Synth with Classical Influence (Psycho-Pass)",
    "Orchestral Drama with Dark Ambience (Tokyo Ghoul)",
    
    # Musical Theater Soundtracks
    "Classic Broadway Show Tunes (The Phantom of the Opera)",
    "Jazz-Inspired Musical with Rhythm and Blues (Chicago)",
    "Orchestral Broadway with Romantic Themes (Les Misérables)",
    "Folk and Acoustic Musical (Once)",
    "Rock Opera with Punk Influence (American Idiot)",
    "Orchestral and Classical Musical (The Lion King)",
    "Pop and Dance Musical with Upbeat Themes (Mamma Mia!)",
    "Classic Broadway with Modern Touches (Hamilton)",
    "Funky Motown and Soul (Motown the Musical)",
    "Contemporary Musical with Pop and Rock Fusion (Dear Evan Hansen)",
    
    # Documentary Soundtracks
    "Ambient Sound Design with Natural Sounds (March of the Penguins)",
    "Minimalist Piano and Orchestral with Nature Elements (Planet Earth II)",
    "Electronic and Atmospheric Soundscapes (Oceans)",
    "Folk Acoustic with Environmental Themes (Won’t You Be My Neighbor?)",
    "Neo-Classical with Global Influences (13th)",
    "Cinematic Orchestral with Global and Cultural Elements (The Last Dance)",
    "Epic Orchestral with Reflective Themes (Free Solo)",
    "Industrial Electro with Nature Soundscapes (The Cove)",
    
    # Classical & Contemporary Music
    "Minimalist Piano with Experimental Elements (Philip Glass)",
    "Baroque-Classical Fusion with Modern Touches (Max Richter)",
    "Cinematic Orchestral with Neo-Classical Influence (Ludovico Einaudi)",
    "Modern Classical with Dark Ambience (Olafur Arnalds)",
    "Post-Rock and Classical Fusion (Explosions in the Sky)",
    "Folk-Classical Hybrid with Contemporary Elements (Sufjan Stevens)",
    
    # Commercial/Advertisement Music
    "Epic Orchestral for Action-Packed Commercials (Nike)",
    "Synthwave Retro-Futuristic with High Energy (Apple iPhone)",
    "Indie Pop and Folk for Emotional Ad Spots (John Lewis Christmas)",
    "Atmospheric Sound Design with Rising Tension (Audi)",
    "Upbeat Pop-Rock and Synth Fusion (Coca-Cola)"


    ]
    
    )

music_genres = list(set(music_genres))

import random
# Step 1: Identify genres containing 'jazz'
jazz_genres = [genre for genre in music_genres if 'jazz' in genre.lower()]

# Step 2: Randomly remove half of the 'jazz' genres
num_to_remove = len(jazz_genres) // 2
jazz_genres_to_remove = random.sample(jazz_genres, num_to_remove)

# Step 3: Remove the selected 'jazz' genres from the original list
music_genres = [genre for genre in music_genres if genre not in jazz_genres_to_remove]
import re
music_genres = [re.sub(r'\(.*?\)', '', genre).strip() for genre in music_genres]

########################################################################
# music_genres = [genre for genre in music_genres if 'indie' in genre.lower()]

# # Step 2: Randomly remove half of the 'jazz' genres
# num_to_remove = len(music_genres) // 2
# jazz_genres_to_remove = random.sample(music_genres, num_to_remove)

# # Step 3: Remove the selected 'jazz' genres from the original list
# music_genres = [genre for genre in music_genres if genre not in jazz_genres_to_remove]
# import re
# music_genres = [re.sub(r'\(.*?\)', '', genre).strip() for genre in music_genres]


from difflib import SequenceMatcher

# Function to calculate the similarity ratio between two strings
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Remove duplicates with high similarity
unique_genres = []
for genre in music_genres:
    if not any(similar(genre, existing) > 0.725 for existing in unique_genres):
        unique_genres.append(genre)

# Now `unique_genres` contains only genres that are not 90% similar
music_genres = unique_genres


# # Step 1: Compute the TF-IDF scores
# vectorizer = TfidfVectorizer(stop_words='english')  # Remove common stop words
# X = vectorizer.fit_transform(music_genres)

# # Step 2: Extract the terms and their corresponding IDF scores
# terms = vectorizer.get_feature_names_out()
# idf_scores = vectorizer.idf_

# # Step 3: Display the terms and their IDF scores
# term_idf = dict(zip(terms, idf_scores))

# # Step 4: Sort terms by their IDF score (low IDF means they appear in many genres)
# sorted_terms = sorted(term_idf.items(), key=lambda x: x[1], reverse=True)

# # Step 5: Optionally, remove terms that have a low IDF score (too frequent)
# threshold = 1.5  # You can adjust this threshold as needed
# filtered_terms = [term for term, idf in sorted_terms if idf >= threshold]

# # Step 6: Rebuild the list with only the filtered terms (smoothing out common words)
# music_genres = [' '.join([word for word in genre.split() if word in filtered_terms]) for genre in music_genres]



print(len(music_genres))

# Define some broad categories
categories = {
    'Electronic': [
        "ambient doom", "acid trance", "psywave", "glitch rap", "future-funk", "vaporwave-funk",
        "dreamcore", "techno-folk fusion", "glitch-hop jazz", "ambient techno", "psy-pop"
    ],
    'Rock': [
        "retro-futuristic rock", "space-funk rock", "space-metal funk", "grunge jazz", "indie-jazz",
        "rock-noir", "grunge-punk", "grungewave"
    ],
    'Folk & Acoustic': [
        "ambient folk", "new-age folk", "tribal-folk", "experimental soul", "indie-folk funk", 
        "space-folk jazz", "freak-folk"
    ],
    'Jazz': [
        "cosmic jazz", "space jazz", "psychedelic soul", "ambient blues", "new wave funk", "jazz-funk-fusion"
    ],
    'Experimental & Avant-Garde': [
        "post-industrial synthwave", "experimental punk", "ambient black metal", "ambient drone", 
        "ambient experimental", "avant-garde blues"
    ],
    'Metal & Doom': [
        "doom jazz", "space doom", "metal-soul", "tribal metal", "doom jazz", "heavy-synth"
    ],
    'Pop & Indie': [
        "dreamwave punk", "indie-synth", "soul-jazz fusion", "pop-metal", "shimmer pop", "indie-hop"
    ],
    'Hip-Hop & Rap': [
        "trap jazz fusion", "metal-hop", "hip-hop metal", "glitch rap", "trapstep", "electro-hop"
    ],
    'Ambient & Chill': [
        "chillwave-funk", "lo-fi fusion", "ambient groove", "chill-metal", "ambient post-punk"
    ]
}


'''
# Step 1: Create a TF-IDF Vectorizer to convert genre names to numerical representations
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(music_genres)

# Step 2: Apply KMeans clustering to group similar genres
num_clusters = 10  # You can change the number of clusters based on your needs
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Step 3: Create a dictionary to map genre to its cluster
genre_clusters = {genre: kmeans.labels_[i] for i, genre in enumerate(music_genres)}

# Step 4: Group genres by their cluster
clustered_genres = {}
for genre, cluster in genre_clusters.items():
    if cluster not in clustered_genres:
        clustered_genres[cluster] = []
    clustered_genres[cluster].append(genre)

# Step 5: Print the number of genres in each cluster
cluster_counts = {cluster: len(genres) for cluster, genres in clustered_genres.items()}
for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster}: {count} genres")

# Step 6: Print genres within each cluster
for cluster, genres in clustered_genres.items():
    print(f"\nCluster {cluster}:")
    print(sorted(genres))
'''




# Step 1: Create a TF-IDF Vectorizer to convert genre names to numerical representations
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(music_genres)

# Step 2: Apply KMeans clustering to group similar genres
num_clusters = 200  # You can change the number of clusters based on your needs
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Step 3: Create a dictionary to map genre to its cluster
genre_clusters = {genre: kmeans.labels_[i] for i, genre in enumerate(music_genres)}

# Step 4: Group genres by their cluster
clustered_genres = {}
for genre, cluster in genre_clusters.items():
    if cluster not in clustered_genres:
        clustered_genres[cluster] = []
    clustered_genres[cluster].append(genre)

# Step 5: Remove genres that are too similar within each cluster
threshold = 0.675  # Set a threshold for similarity (0.85 means 85% similarity)

filtered_clustered_genres = {}
for cluster, genres in clustered_genres.items():
    # Get the indices of the genres in the current cluster
    genre_indices = [music_genres.index(genre) for genre in genres]
    genre_matrix = X[genre_indices]
    
    # Compute pairwise cosine similarities
    similarity_matrix = cosine_similarity(genre_matrix)
    
    # Create a list to keep track of genres to remove
    to_remove = set()
    
    # Iterate through all pairs of genres in the cluster
    for i, genre_i in enumerate(genres):
        for j in range(i + 1, len(genres)):
            if similarity_matrix[i, j] > threshold:  # If the similarity is too high
                # Mark the second genre as too similar (to remove it)
                to_remove.add(genres[j])
    
    # Filter out genres that are too similar
    filtered_clustered_genres[cluster] = [genre for genre in genres if genre not in to_remove]


# Step 6: Create a new list ordered by cluster, picking 80% of genres randomly from clusters larger than 10 genres
ordered_genres = []
for cluster in sorted(filtered_clustered_genres.keys()):
    # Check if the cluster has more than 10 genres
    if len(filtered_clustered_genres[cluster]) > 100:
        # Calculate the number of genres to pick (80% of the total)
        num_genres_to_pick = int(len(filtered_clustered_genres[cluster]) * 0.6)
        
        # Pick 80% of the genres randomly
        sampled_genres = random.sample(filtered_clustered_genres[cluster], num_genres_to_pick)
        
        # Add the selected genres to the ordered list
        ordered_genres.extend(sorted(sampled_genres))
    else:
        # If the cluster has 10 or fewer genres, add all genres to the ordered list
        ordered_genres.extend(sorted(filtered_clustered_genres[cluster]))

# Step 7: Print the number of genres in each filtered cluster
print("\nNumber of genres in each filtered cluster:")
c = 0
for cluster, genres in filtered_clustered_genres.items():
    print(f"Cluster {cluster}: {len(genres)} genres")
    c+=1
    if c>100:
        break

# Print the result
print("\nFiltered genres ordered by cluster:")
for cluster, genres in filtered_clustered_genres.items():
    print(f"\nCluster {cluster}:")
    from pathlib import Path
    num_files = len([f for f in Path('database/urls').iterdir() if f.is_file()])
    print(sorted(genres)[0:25])  # You can print the top 3 genres from each cluster for a preview


# print("\nOrdered list of genres:")
# print(ordered_genres)

print("\nOrdered list of genres:")
# print(ordered_genres)

import pickle
# Step 1: Open a file in write-binary mode
# with open(r'src\fast\preprocessing\webscrape\lists_and_save_files\ordered_genres2.pkl', 'wb') as file:
#     # Step 2: Serialize the list and save it to the file
#     pickle.dump(ordered_genres, file)

# with open(r'src\fast\preprocessing\webscrape\lists_and_save_files\ordered_genres.pkl', 'rb') as file:
#     # Step 2: Serialize the list and save it to the file
#     ordered_genres2 = pickle.load(file)

print(len(ordered_genres))
print("List saved successfully.")


# print(len(ordered_genres2))
# print("List saved successfully.")