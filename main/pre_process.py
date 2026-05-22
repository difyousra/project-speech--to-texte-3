import re 

def pre_process(text:str) -> str :
    if not isinstance(text, str):
        return ""

    # definir les pattern 
    URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
    EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
    USERNAME_PATTERN = re.compile(r"(?<!\w)@\w+")
    DATE_FULL_PATTERN = re.compile(r"(?<!\d)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})(?!\d)")
    DATE_AMBIG_PATTERN = re.compile(r"(?<!\d)(\d{1,2}/\d{1,2})(?!\d)")
    CLAIM_ID_PATTERN = re.compile(r"(?<!\d)(\d{16})(?!\d)")
    SN_PATTERN = re.compile(r"(?<!\d)(\d{20})(?!\d)")
    PHONE_PATTERN = re.compile(r"(?<!\w)(\+?\d[\d\s\-\(\)]{5,}\d)(?!\w)")
    SPECIAL_NUMBERS = {"12", "100", "1500"}
    ALGERIE_TELECOM_PATTERN = re.compile(r"Algérie Télécom\s*-\s*", re.IGNORECASE)
    REMOVE_PATTERN = re.compile(r'[\(\)"#\[\]\*]')
    MULTI_SPACE_PATTERN = re.compile(r"\s+")


    # fonction intermediaire 
    def replace_phone(text):
        def repl(match):
            candidate = match.group(1)
            digits = re.sub(r"\D", "", candidate)

            if digits in SPECIAL_NUMBERS:
                return candidate

            if 9 <= len(digits) <= 12:
                return "<PHONE>"

            return candidate

        return PHONE_PATTERN.sub(repl, text)

    # appliquer les patterns si detecter 
    text = URL_PATTERN.sub("<URL>", text)
    text = EMAIL_PATTERN.sub("<EMAIL>", text)
    text = USERNAME_PATTERN.sub("USERNAME", text)
    text = SN_PATTERN.sub("<SN>", text)
    text = CLAIM_ID_PATTERN.sub("<CLAIM_ID>", text)
    text = DATE_FULL_PATTERN.sub("<DATE>", text)
    text = replace_phone(text)
    text = ALGERIE_TELECOM_PATTERN.sub("", text)
    text = REMOVE_PATTERN.sub("", text)


    # nettoyer les espaces apres avoir appliquer 
    clean_text = MULTI_SPACE_PATTERN.sub(" ", text).strip()

    return clean_text