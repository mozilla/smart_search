from recognizers_text import Culture
from recognizers_date_time import DateTimeRecognizer
from datetime import datetime
import dateparser
import re
import rapidfuzz
import calendar
import num2words

def day_bounds(d):
    """Clamp a datetime to its calendar day."""
    start = d.replace(hour=0, minute=0, second=0, microsecond=0)
    end   = d.replace(hour=23, minute=59, second=59, microsecond=999_000)
    return start, end


def get_dt_range(text, current_dt, lang="en"):
    """
    In recognizers-text, the parsed result called resolution
    it could have multiple results/resolutions, for example "show me news from yesterday and today"
    we always return one resolution only, here are the rules:
    1. loop through resolutions, return the first range we found
    2. if no range result found, loop again and find the first single date point result
    3. return None
    """

    if lang == "en":
        culture = Culture.English
        model = DateTimeRecognizer(culture).get_datetime_model()
    else:
        print(f"Language {lang} is not supported.")
        return

    resolutions = model.parse(text, current_dt)

    for r in resolutions:
        values = r.resolution.get("values", [])
        for v in values:
            type_, timex = v.get("type"), v.get("timex")
            # # some range cases may not have "end", e.g. "since"
            if type_ in ("daterange", "datetimerange") and (v.get("start") or v.get("end")):
                # start = datetime.strptime(v.get("start"), "%Y-%m-%d")
                # end = datetime.strptime(v.get("end"), "%Y-%m-%d")
                # it could be date or datetime "2025-09-25", "2025-09-25 17:00:00"
                # use dateparser for convenience for now
                if v.get("start"):
                    start = dateparser.parse(v["start"])
                else:
                    start = dateparser.parse("1980-01-01")
                # mod values:
                # https://github.com/microsoft/Recognizers-Text/blob/master/JavaScript/packages/recognizers-date-time/src/dateTime/constants.ts#L56-L76
                if v.get("end"):
                    end = dateparser.parse(v["end"])
                else:
                    end = current_dt
                    if type(current_dt) == str:
                        end = dateparser.parse(current_dt)
                #
                # if v.get("Mod") and v.get("Mod") in ("before", "until", "less", "end"):
                #     start, end = datetime.min, start


                return day_bounds(start)[0], day_bounds(end)[1]


    for r in resolutions:
        values = r.resolution.get("values", [])
        for v in values:
            type_, timex = v.get("type"), v.get("timex")
            if type_ in ("date", "datetime") and v.get("value"):
                start = dateparser.parse(v["value"])
                if start:
                    return day_bounds(start)[0], day_bounds(start)[1]

    return


def build_vocab(lang="en"):
    """
    Returns a set of words (strings) that are relevant to temporal expressions.
    This is for gather all words that related to our temporal task, for fuzzy typo.
    For now English only
    """

    base = {
        "yesterday", "today", "tomorrow", "tonight", "now",
        "last", "next", "ago", "in", "since", "after", "before",
        "between", "to", "until", "through", "till", "on", "at", "around", "about",
        "second", "seconds", "minute", "minutes", "hour", "hours",
        "day", "days", "week", "weeks", "month", "months", "year", "years",
        "morning", "afternoon", "evening", "night",
    }

    vocab = set(base)

    # handles weekdays/months (for now we only handle English)
    # for other languages, could use Babel (in JS/C++ should has built-in to handle it)
    vocab.update([d.lower() for d in calendar.day_name])
    vocab.update([m.lower() for m in calendar.month_name if m])

    # words of 0-120, 120 is kind of like sweet spot, for more than 120, people usually type in number
    # for example: 365 days, not much people will type three hundred.... days
    for n in range(0, 121):
        w = num2words.num2words(n, lang=lang)
        for token in re.split(r"[\s\-]+", w): # it returns like "twenty-one"
            if token:
                vocab.add(token.lower())

    # ordinal words: for 21 return twenty-first
    for n in range(1, 32):
        w = num2words.num2words(n, to="ordinal", lang=lang)
        for token in re.split(r"[\s\-]+", w):
            if token:
                vocab.add(token.lower())

    return vocab

ABBR_MAP = {
    # day
    "mon": "monday", "tue": "tuesday", "tues": "tuesday",
    "wed": "wednesday", "weds": "wednesday",
    "thu": "thursday", "thur": "thursday", "thurs": "thursday",
    "fri": "friday", "sat": "saturday", "sun": "sunday",
    # month
    "jan": "january", "feb": "february", "mar": "march", "apr": "april", "may": "may",
    "jun": "june", "jul": "july", "aug": "august",
    "sep": "september", "sept": "september",
    "oct": "october", "nov": "november", "dec": "december",
}


def normalize_query(query, lang="en", threshold=80):
    """
    It loop through all alphabetic words in text, compare each word to the whole lexicon to find the similarity word.
    Use the threshold to deside replace it or not.
    It also has C++/JS version, but if not, we could also implement one, as just comparison, and our lex is very small.
    Lowercase + OPTIONAL fuzzy typo correction against a generated temporal lexicon.
    - 'lang' localizes weekdays/months/number-words if Babel/num2words present
    - Uses rapidfuzz if available; otherwise returns lowercase text unchanged
    - Only repairs alphabetic tokens of length >=3; digits/spacing/punctuation preserved
    """

    query = query.lower()

    vocab = build_vocab(lang=lang)

    # split the query into parts of alphabetic and non-alphabetic
    tokens = re.findall(r"\w+|\W+", query)

    res = []

    for token in tokens:
        if token.isalpha() and len(token) >= 3:

            if token in ABBR_MAP:
                res.append(ABBR_MAP[token])
                continue

            best = rapidfuzz.process.extractOne(token, vocab, scorer=rapidfuzz.fuzz.QRatio)
            if best and best[1] >= threshold:
                res.append(best[0])
            else:
                res.append(token)
        else:
            res.append(token)

    return "".join(res)


def parse_time_window(query, current_dt, lang="en"):
    """
    """

    # lower() + fix typo
    query = normalize_query(query)
    if type(current_dt) == str:
        current_dt = datetime.strptime(current_dt, "%Y-%m-%d")
    r = get_dt_range(query, current_dt, lang=lang)
    return r