"""Shared utilities."""
import re

_LANG_MARKERS: dict[str, frozenset[str]] = {
    "Italian": frozenset({
        "il","la","lo","gli","le","un","una","di","del","della","dei","degli","delle",
        "che","è","non","con","per","si","come","dove","quando","questo","questa",
        "questi","queste","quale","quali","sono","hai","siamo","avete","hanno",
        "cosa","perché","però","quindi","anche","già","ancora","sempre","mai",
        "tutto","tutti","tutta","tutte","fare","dire","vedere","sapere","volere",
    }),
    "French": frozenset({
        "le","la","les","un","une","des","je","tu","il","elle","nous","vous","ils",
        "est","sont","que","pas","pour","avec","comme","où","quand","cette","quel",
        "aussi","très","bien","mais","donc","tout","tous","toute","toutes",
        "être","avoir","faire","dire","voir","savoir","vouloir","pouvoir","aller",
    }),
    "German": frozenset({
        "der","die","das","des","dem","den","ein","eine","ist","sind","war","waren",
        "nicht","mit","für","von","bei","nach","aus","an","auf","über","unter",
        "ich","du","er","sie","wir","auch","noch","schon","immer","alles","alle",
        "sein","haben","werden","können","müssen","sollen","wollen","machen","sagen",
    }),
    "Spanish": frozenset({
        "el","la","los","las","un","una","es","son","fue","fueron","que","no","con",
        "para","por","como","donde","cuando","este","esta","estos","estas","cual",
        "también","muy","bien","pero","porque","todo","todos","toda","todas",
        "ser","tener","hacer","decir","ver","saber","querer","poder","ir","venir",
    }),
    "Portuguese": frozenset({
        "o","a","os","as","um","uma","é","são","foi","foram","que","não","com",
        "para","por","como","onde","quando","este","esta","estes","estas","qual",
        "também","muito","bem","mas","porque","todo","todos","toda","todas",
        "ser","ter","fazer","dizer","ver","saber","querer","poder","ir","vir",
    }),
}


def detect_language(text: str) -> str:
    """Detect language from text using stopword matching. Defaults to English."""
    words = frozenset(re.findall(r'\b\w+\b', text.lower()))
    best_lang, best_score = "English", 0
    for lang, markers in _LANG_MARKERS.items():
        score = len(words & markers)
        if score > best_score:
            best_lang, best_score = lang, score
    return best_lang
