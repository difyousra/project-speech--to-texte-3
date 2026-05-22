import re


def pre_process(text: str) -> str:
    if not isinstance(text, str):
        return ""

    url_pattern = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
    email_pattern = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
    username_pattern = re.compile(r"(?<!\w)@\w+")
    date_full_pattern = re.compile(r"(?<!\d)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})(?!\d)")
    claim_id_pattern = re.compile(r"(?<!\d)(\d{16})(?!\d)")
    sn_pattern = re.compile(r"(?<!\d)(\d{20})(?!\d)")
    phone_pattern = re.compile(r"(?<!\w)(\+?\d[\d\s\-\(\)]{5,}\d)(?!\w)")
    special_numbers = {"12", "100", "1500"}
    algerie_telecom_pattern = re.compile(r"Algérie Télécom\s*-\s*", re.IGNORECASE)
    remove_pattern = re.compile(r'[\(\)"#\[\]\*]')
    multi_space_pattern = re.compile(r"\s+")

    def replace_phone(value: str) -> str:
        def repl(match):
            candidate = match.group(1)
            digits = re.sub(r"\D", "", candidate)
            if digits in special_numbers:
                return candidate
            if 9 <= len(digits) <= 12:
                return "<PHONE>"
            return candidate

        return phone_pattern.sub(repl, value)

    text = url_pattern.sub("<URL>", text)
    text = email_pattern.sub("<EMAIL>", text)
    text = username_pattern.sub("USERNAME", text)
    text = sn_pattern.sub("<SN>", text)
    text = claim_id_pattern.sub("<CLAIM_ID>", text)
    text = date_full_pattern.sub("<DATE>", text)
    text = replace_phone(text)
    text = algerie_telecom_pattern.sub("", text)
    text = remove_pattern.sub("", text)
    return multi_space_pattern.sub(" ", text).strip()
