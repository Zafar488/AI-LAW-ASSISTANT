"""Advanced Legal Retriever for Pakistan Statutory Knowledge Base.

Combines:
1. Dense semantic search using FAISS (Cosine Similarity on L2-normalized embeddings)
2. Comprehensive Statutory Catalog covering all 72+ Pakistan law books
3. Multi-lingual legal query parsing (English, Urdu, Roman Urdu, Sections, Articles, Orders, Rules)
4. Dynamic candidate pool expansion with exact statutory section & order injection
5. Hybrid scoring fusion (Semantic similarity + Statutory alignment + Exact section match + Token overlap)
6. Diverse passage deduplication and citation formatting
"""

from __future__ import annotations

import json
import os
import pickle
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    BASE_DIR,
    EMBEDDINGS_DIR,
    EMBEDDING_MODEL,
)

INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.pkl"
MANIFEST_PATH = EMBEDDINGS_DIR / "index_manifest.json"

URDU_DIGITS_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

ROMAN_TO_INT: dict[str, int] = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
    "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17, "xviii": 18, "xix": 19, "xx": 20,
    "xxi": 21, "xxii": 22, "xxiii": 23, "xxiv": 24, "xxv": 25, "xxvi": 26, "xxvii": 27, "xxviii": 28, "xxix": 29, "xxx": 30,
    "xxxi": 31, "xxxii": 32, "xxxiii": 33, "xxxiv": 34, "xxxv": 35, "xxxvi": 36, "xxxvii": 37, "xxxviii": 38, "xxxix": 39, "xl": 40,
    "xli": 41, "xlii": 42, "xliii": 43, "xliv": 44, "xlv": 45, "xlvi": 46, "xlvii": 47, "xlviii": 48, "xlix": 49, "l": 50
}
INT_TO_ROMAN: dict[int, str] = {v: k.upper() for k, v in ROMAN_TO_INT.items()}

# Regex to detect Section, Article, Rule, and standalone section citations
SECTION_QUERY_REGEX = re.compile(
    r"(?:\b(?:section|sec\.?|s\.|article|art\.?)|(?:دفعہ|آرٹیکل|سیکشن))\s*(\d+(?:-[A-Za-z\d]+|[A-Za-z])?|[۰-۹]+)",
    re.IGNORECASE,
)
STANDALONE_SECTION_REGEX = re.compile(
    r"\b(\d{1,4}(?:-[A-Za-z\d]+|[A-Za-z]))\b",
    re.IGNORECASE,
)
ORDER_RULE_REGEX = re.compile(
    r"(?:\border|آرڈر)\s*([IVXLCDM\d]+|[۰-۹]+)(?:\s*(?:rule|قاعدہ|رول)\s*(\d+[A-Za-z]?|[۰-۹]+))?",
    re.IGNORECASE,
)

# Comprehensive catalog covering all 72+ statutes and legal books in Pakistan legal knowledge base
STATUTE_CATALOG: list[dict[str, Any]] = [
    {
        "name": "Pakistan Penal Code",
        "title_patterns": ["pakistan penal code", "penal code"],
        "keywords": [
            "ppc", "penal code", "pakistan penal", "ta'zir", "tazir", "hadd", "qisas", "diyat", "qatl", "murder",
            "302", "hurt", "337", "kidnapping", "abduction", "365", "rape", "376", "theft", "379", "robbery", "392",
            "dacoity", "395", "cheating", "fraud", "420", "dishonoured cheque", "dishonored cheque", "check bounce",
            "cheque bounce", "489-f", "489f", "defamation", "499", "500", "harassment of woman", "509", "criminal trespass",
            "441", "447", "criminal intimidation", "506", "unlawful assembly", "141", "149", "188", "disobedience to order",
            "sedition", "124-a", "false evidence", "perjury", "193", "criminal breach of trust", "406", "forgery", "468", "471",
            "تعزیرات پاکستان", "مجموعہ تعزیرات", "قتل", "دھوکہ دہی", "چیک باؤنس", "زنا", "سزا", "ہراسانی", "چوری", "ڈکیتی",
            "خیانت", "تلبیس", "جعلسازی", "امانت میں خیانت"
        ],
    },
    {
        "name": "Code of Criminal Procedure",
        "title_patterns": ["criminal procedure", "cr.p.c", "crpc"],
        "keywords": [
            "crpc", "cr.p.c", "criminal procedure", "fir", "first information report", "154", "police investigation",
            "156", "statement", "161", "challan", "police report", "173", "remand", "physical remand", "judicial remand",
            "167", "bail", "pre-arrest bail", "post-arrest bail", "protective bail", "anticipatory bail", "497", "498", "498-a",
            "section 144", "144", "justice of the peace", "ex-officio justice of the peace", "22-a", "22a", "22-b", "22b",
            "quashment", "quashment of fir", "quash", "561-a", "561a", "inherent powers", "acquittal", "early acquittal",
            "249-a", "249a", "265-k", "265k", "charge", "trial", "sessions court", "magistrate", "summary trial", "appeal",
            "revision", "435", "439", "warrant", "summons", "younas abbas",
            "ضابطہ فوجداری", "مجموعہ ضابطہ فوجداری", "ایف آئی آر", "ضمانت", "قبل از گرفتاری", "بعد از گرفتاری",
            "حفاظتی ضمانت", "ریمانڈ", "چالان", "دفعہ 144", "اخراج مقدمہ", "اخراج ایف آئی آر", "کوئیشمنٹ",
            "جسٹس آف دی پیس", "برائیت", "عدالتی طریقہ کار کا ناجائز استعمال"
        ],
    },
    {
        "name": "Code of Civil Procedure",
        "title_patterns": ["civil procedure", "c.p.c", "cpc"],
        "keywords": [
            "cpc", "c.p.c", "civil procedure", "plaint", "written statement", "suit", "order 39", "order xxxix",
            "order 39 rule 1", "order 39 rule 2", "order 39 rule 2(3)", "order 39 rule 2-a", "temporary injunction",
            "stay order", "interim stay", "injunction", "disobedience of injunction", "breach of stay", "civil prison",
            "attachment of property", "status quo", "status quo ante", "restoration of possession", "section 151", "151 cpc",
            "inherent powers civil", "order 7 rule 11", "rejection of plaint", "res judicata", "section 11", "execution of decree",
            "order 21", "first appeal", "second appeal", "section 96", "section 100", "civil revision", "section 115",
            "review", "order 47", "ex-parte decree", "order 9 rule 13", "haji baha-ud-din", "bakhtawar", "fruits of contempt",
            "ضابطہ دیوانی", "مجموعہ ضابطہ دیوانی", "دعویٰ", "جواب دعویٰ", "سٹے آرڈر", "حکم امتناعی", "حکم امتناع",
            "عارضی حکم امتناعی", "سٹے آرڈر کی خلاف ورزی", "حکم امتناعی کی خلاف ورزی", "سابقہ پوزیشن کی بحالی",
            "سول جیل", "جائیداد کی قرقی", "قبضہ کی واپسی", "خارج دعویٰ", "حکم امتناع کی منسوخی"
        ],
    },
    {
        "name": "Constitution of Pakistan",
        "title_patterns": ["constitution"],
        "keywords": [
            "constitution", "constitutional", "fundamental rights", "article 199", "writ", "writ petition", "mandamus",
            "certiorari", "habeas corpus", "quo warranto", "prohibition", "article 184(3)", "original jurisdiction", "suo motu",
            "article 9", "security of person", "article 10", "safeguards arrest", "article 10a", "fair trial", "article 14",
            "dignity of man", "article 15", "freedom of movement", "article 16", "freedom of assembly", "article 17",
            "freedom of association", "article 18", "freedom of trade", "article 19", "freedom of speech", "article 19a",
            "right to information", "article 20", "religious freedom", "article 25", "equality", "article 25a", "education",
            "article 204", "contempt of court", "article 270", "supreme court", "high court",
            "آئین", "دستور", "آئین پاکستان", "دستور پاکستان", "بنیادی حقوق", "آزادی اظہار", "پرامن احتجاج",
            "لانگ مارچ", "جلسہ", "جلوس", "احتجاج", "دھرنا", "مارچ", "اسمبلی", "اجتماع", "ریٹ", "سپریم کورٹ", "ہائی کورٹ", "توہین عدالت"
        ],
    },
    {
        "name": "Qanun-e-Shahadat",
        "title_patterns": ["qanun", "shahadat"],
        "keywords": [
            "qso", "qanun", "shahadat", "qanun-e-shahadat", "evidence", "witness", "eyewitness", "admission",
            "confession", "dying declaration", "burden of proof", "documentary evidence", "primary evidence",
            "secondary evidence", "expert opinion", "cross-examination", "examination-in-chief", "competence of witness",
            "accomplice", "electronic evidence", "article 164",
            "قانون شہادت", "گواہی", "گواہ", "اعتراف", "ثبوت", "بار ثبوت", "دستاویزی ثبوت", "شہادت", "جرح"
        ],
    },
    {
        "name": "Family Courts Act",
        "title_patterns": ["family courts act", "family court"],
        "keywords": [
            "family court", "family courts act", "khula", "maintenance", "dower", "mahr", "custody", "visitation",
            "restitution of conjugal rights", "section 5 family", "section 9 family", "section 14 family appeal",
            "pre-trial", "post-trial",
            "فیملی کورٹ", "فیملی کورٹس ایکٹ", "خلع", "نان نفقہ", "حق مہر", "خرچہ", "حضانت", "طلاق", "بچوں کی تحویل", "ملاقات بچے"
        ],
    },
    {
        "name": "Muslim Family Laws Ordinance",
        "title_patterns": ["muslim family laws", "muslim family"],
        "keywords": [
            "muslim family", "mflo", "muslim family laws ordinance", "talaq", "divorce", "arbitration council",
            "iddat", "polygamy", "second marriage", "section 4 mflo", "section 5 mflo", "section 6 mflo", "section 7 mflo",
            "section 9 maintenance", "nikahnama",
            "مسلم عائلی قوانین", "ثالثی کونسل", "عدت", "دوسری شادی", "طلاق کا نوٹس", "رجسٹریشن نکاح"
        ],
    },
    {
        "name": "Dissolution of Muslim Marriages Act",
        "title_patterns": ["dissolution of muslim marriages"],
        "keywords": [
            "dissolution of muslim marriages", "dissolution of marriage", "grounds for divorce", "cruelty",
            "non-maintenance", "impotency", "insanity", "missing husband", "failure to perform marital obligations",
            "تنسیخ نکاح", "تفریق", "نکاح کی منسوخی"
        ],
    },
    {
        "name": "Guardians and Wards Act",
        "title_patterns": ["guardians and wards"],
        "keywords": [
            "guardians and wards", "guardian", "ward", "custody of minor", "welfare of minor", "guardianship certificate",
            "section 25 custody", "section 12 custody",
            "بچوں کی سرپرستی", "سرپرستی", "ولی", "کفالت", "بچے کی تحویل"
        ],
    },
    {
        "name": "Child Marriage Restraint Acts",
        "title_patterns": ["child marriage", "child marriages"],
        "keywords": [
            "child marriage", "underage", "minor marriage", "child marriage restraint act",
            "punjab child marriage restraint act 2026", "sindh child marriages restraint act", "ict child marriage",
            "minimum age 18", "nikah khawan liability",
            "کم عمری کی شادی", "چائلڈ میرج", "نابالغ کی شادی", "کم عمر نکاح"
        ],
    },
    {
        "name": "Dowry and Bridal Gifts (Restriction) Act",
        "title_patterns": ["dowry and bridal gifts"],
        "keywords": [
            "dowry", "bridal gifts", "dowry act", "dowry restriction", "recovery of dowry", "jahez", "list of dowry articles",
            "جہیز", "سامان جہیز", "جہیز کی واپسی", "بری"
        ],
    },
    {
        "name": "Protection Against Harassment of Women at the Workplace Act",
        "title_patterns": ["protection against harassment of women at the workplace"],
        "keywords": [
            "harassment", "workplace harassment", "sexual harassment", "protection against harassment",
            "inquiry committee", "internal inquiry committee", "ombudsman", "ombudsperson", "fospah", "code of conduct",
            "competent authority", "section 3 harassment", "section 4 inquiry", "section 8 ombudsman", "section 9 appeal",
            "hostile work environment",
            "ہراساں", "ہراسانی", "ورک پلیس", "کام کی جگہ", "دفتر میں ہراساں", "محتسب", "انکوائری کمیٹی", "فوسپاہ", "جنسی ہراسانی"
        ],
    },
    {
        "name": "Protection of Women (Criminal Laws Amendment Act) 2006",
        "title_patterns": ["women rights", "protection of women criminal laws"],
        "keywords": [
            "protection of women act 2006", "hudood", "zina", "qazf", "offence of zina", "rape vs zina", "hudood laws amendment",
            "تحفظ نسواں ایکٹ", "حدود آرڈیننس", "زنا آرڈیننس"
        ],
    },
    {
        "name": "Acid Control and Acid Crime Prevention Act",
        "title_patterns": ["acid control"],
        "keywords": [
            "acid control", "acid crime", "acid attack", "acid throwing", "section 336a", "section 336b", "corrosive substance",
            "تیزاب پھینکنا", "تیزاب گردی", "تیزاب کا حملہ"
        ],
    },
    {
        "name": "Juvenile Justice System Act",
        "title_patterns": ["juvenile justice"],
        "keywords": [
            "juvenile justice", "jjsa", "juvenile court", "juvenile offender", "child offender", "minor accused",
            "juvenile rehabilitation", "juvenile bail", "determination of age",
            "نوعمر ملزم", "کم عمر ملزم", "جووینائل کورٹ"
        ],
    },
    {
        "name": "Illegal Dispossession Act",
        "title_patterns": ["illegal dispossession"],
        "keywords": [
            "illegal dispossession", "land grabbing", "qabza", "dispossessed", "section 3 illegal dispossession",
            "act xi of 2005", "section 4 complaint", "section 7 interim possession", "section 8 eviction", "land mafia",
            "property occupant",
            "غیر قانونی قبضہ", "قبضہ مافیا", "بے دخلی", "قبضہ واگزار"
        ],
    },
    {
        "name": "Specific Relief Act",
        "title_patterns": ["specific performance", "specific relief"],
        "keywords": [
            "specific relief", "specific performance", "suit for declaration", "section 42 sra", "perpetual injunction",
            "section 54 sra", "mandatory injunction", "section 55 sra", "possession immovable property", "section 8 sra",
            "section 9 sra", "cancellation of instruments", "section 39 sra",
            "داد رسی", "تکمیل مختص", "استقرار حق", "حکم امتناعی دوامی", "منسوخی دستاویز"
        ],
    },
    {
        "name": "Transfer of Property Act",
        "title_patterns": ["transfer of property"],
        "keywords": [
            "transfer of property", "tpa", "sale deed", "mortgage", "lease", "gift deed", "hiba", "lis pendens",
            "section 52 tpa", "part performance", "section 53a tpa", "exchange",
            "انتقال جائیداد", "بیع نامہ", "رہن", "ہبہ", "پٹہ", "کرایہ داری"
        ],
    },
    {
        "name": "Easements Act",
        "title_patterns": ["easements act"],
        "keywords": [
            "easements act", "easement", "right of way", "right of light", "right of air", "customary easement",
            "prescriptive easement", "license",
            "حق آسائش", "حق ارتفاق", "راستہ کا حق"
        ],
    },
    {
        "name": "Contract Act",
        "title_patterns": ["contact act", "contract act"],
        "keywords": [
            "contract", "agreement", "consideration", "void", "breach", "damages", "section 73 contract",
            "section 74 contract", "free consent", "coercion", "undue influence", "fraud", "misrepresentation",
            "indemnity", "guarantee", "bailment", "agency",
            "معاہدہ", "قانون معاہدہ", "معاہدے کی خلاف ورزی", "بدل", "ہرجانہ"
        ],
    },
    {
        "name": "Sale of Goods Act",
        "title_patterns": ["sale of goods"],
        "keywords": [
            "sale of goods", "buyer", "seller", "unpaid seller", "conditions and warranties", "delivery of goods",
            "rejection of goods",
            "بیع اشیاء", "سامان کی فروخت"
        ],
    },
    {
        "name": "Limitation Act",
        "title_patterns": ["limitation act"],
        "keywords": [
            "limitation", "time barred", "prescribed period", "condonation", "section 3 limitation", "section 5 limitation",
            "section 12 limitation", "first schedule",
            "قانون میعاد", "میعاد سماعت", "زائد المیعاد", "تاخیر کی معافی"
        ],
    },
    {
        "name": "Arbitration Act",
        "title_patterns": ["arbitration act"],
        "keywords": [
            "arbitration", "arbitrator", "award", "arbitration agreement", "umpire", "setting aside award",
            "section 30 arbitration", "section 34 stay",
            "قانون ثالثی", "ثالثی", "ثالث", "فیصلہ ثالثی", "حکم ثالثی"
        ],
    },
    {
        "name": "Prevention of Electronic Crimes Act",
        "title_patterns": ["peca", "prevention of electronic crimes"],
        "keywords": [
            "peca", "cyber", "electronic crime", "cyber crime", "harassment online", "defamation online",
            "unauthorized access", "section 3 peca", "cyber terrorism", "section 10 peca", "electronic fraud",
            "section 14 peca", "identity theft", "section 16 peca", "cyber stalking", "section 24 peca",
            "fia cyber crime",
            "پیکا", "سائبر کرائم", "آن لائن ہراسانی", "الیکٹرانک کرائمز", "بلیک میلنگ", "شناختی معلومات کا غلط استعمال"
        ],
    },
    {
        "name": "National Accountability Ordinance",
        "title_patterns": ["national accountability ordinance", "nao"],
        "keywords": [
            "nao", "nab", "accountability", "corruption", "assets beyond means", "corrupt practices", "section 9 nao",
            "plea bargain", "section 25 nao", "voluntary return", "accountability court", "nab remand", "nab bail",
            "نیب", "احتساب", "کرپشن", "بدعنوانی", "آمدن سے زائد اثاثے", "پلی بارگین"
        ],
    },
    {
        "name": "Anti-Terrorism Act",
        "title_patterns": ["anti terrorism", "anti-terrorism"],
        "keywords": [
            "ata", "anti terrorism", "terrorist", "7 ata", "section 6 ata", "section 7 ata", "anti terrorism court",
            "atc", "fourth schedule", "4th schedule", "banned organization", "proscribed organization", "extortion", "sectarian",
            "انسداد دہشت گردی", "دہشت گردی", "فورتھ شیڈول", "کالعدم تنظیم"
        ],
    },
    {
        "name": "Anti-Narcotics Force & Drugs Act",
        "title_patterns": ["anti narcotics force", "drugs act"],
        "keywords": [
            "anf", "narcotics", "anti narcotics", "cnsa", "control of narcotic substances", "drugs act", "charras",
            "heroin", "ice", "methamphetamine", "opium", "section 9 cnsa", "9c death penalty", "spurious drugs",
            "منشیات", "انسداد منشیات", "چرس", "ہیروئن", "منشیات فروش"
        ],
    },
    {
        "name": "Companies Act",
        "title_patterns": ["companies act"],
        "keywords": [
            "companies act", "company", "secp", "directors", "shares", "incorporation", "board of directors", "agm",
            "shareholders", "winding up", "liquidation", "oppression", "mismanagement",
            "کمپنیز ایکٹ", "کمپنی", "ایس ای سی پی", "ڈائریکٹرز", "حصص", "شیئر ہولڈر"
        ],
    },
    {
        "name": "Police Order and Police Rules",
        "title_patterns": ["police order", "police act", "police rules", "punjab police rules", "sindah police rules"],
        "keywords": [
            "police order", "police act", "police rules", "sho", "dpo", "cpo", "ccpo", "igp", "ppo", "police complaint",
            "public safety commission", "police diary", "roznamcha", "arrest without warrant",
            "پولیس آرڈر", "پولیس رولز", "ایس ایچ او", "ڈی پی او", "پولیس کے خلاف شکایت", "روزنامچہ", "تفتیشی افسر"
        ],
    },
    {
        "name": "Rules of Business",
        "title_patterns": ["rules of business"],
        "keywords": [
            "rules of business", "punjab government rules of business", "sindh government rules of business",
            "kpk rules of business", "balochistan rules of business", "secretariat instructions", "cabinet procedure",
            "allocation of business", "transaction of business",
            "قواعد و ضوابط", "رولز آف بزنس", "سیکرٹریٹ", "کابینہ"
        ],
    },
    {
        "name": "Legal Practitioners and Bar Councils Act",
        "title_patterns": ["legal and practioner", "legal practitioners"],
        "keywords": [
            "legal practitioners", "bar council", "advocate", "advocate high court", "advocate supreme court",
            "pakistan bar council", "provincial bar council", "professional misconduct", "right to practice",
            "وکلاء", "بار کونسل", "وکالت", "پیشہ ورانہ ضابطہ اخلاق"
        ],
    },
    {
        "name": "Probation of Offenders Ordinance",
        "title_patterns": ["probation of offenders"],
        "keywords": [
            "probation of offenders", "probation", "probation officer", "conditional discharge", "first offender",
            "پربیشن", "نیک چلنی", "رہائی بر بنائے نیک چلنی"
        ],
    },
    {
        "name": "Security of Pakistan Act",
        "title_patterns": ["security of pakistan"],
        "keywords": [
            "security of pakistan", "preventive detention", "state security", "public safety",
            "سلامتی پاکستان", "حفاظتی نظر بندی"
        ],
    },
    {
        "name": "Asaan Karobar Act",
        "title_patterns": ["asaan karobar"],
        "keywords": [
            "asaan karobar", "ease of doing business", "regulatory simplification", "business deregulation",
            "آسان کاروبار"
        ],
    },
    {
        "name": "Abandoned Properties Act",
        "title_patterns": ["abandoned properties"],
        "keywords": [
            "abandoned properties", "custodian abandoned property", "evacuee property",
            "متروکہ املاک", "متروکہ جائیداد"
        ],
    },
    {
        "name": "Armed Forces Transport & Dangerous Cargoes",
        "title_patterns": ["armed forces civil general transport", "dangerous cargoes"],
        "keywords": [
            "armed forces transport", "requisition of transport", "dangerous cargoes", "explosives", "hazardous cargo"
        ],
    },
    {
        "name": "Electricity Control Ordinance",
        "title_patterns": ["electricity control"],
        "keywords": [
            "electricity control", "electric power", "wapda", "power supply", "کنٹرول بجلی"
        ],
    },
    {
        "name": "Traffic Offences Ordinance",
        "title_patterns": ["traffic offences"],
        "keywords": [
            "traffic offences", "special court traffic", "motor vehicles", "road accidents",
            "ٹریفک چالان", "ٹریفک کورٹ"
        ],
    },
    {
        "name": "Aga Khan Properties Succession and Transfer Act",
        "title_patterns": ["aga khan properties"],
        "keywords": [
            "aga khan properties", "succession", "transfer of property aga khan"
        ],
    },
    {
        "name": "Akhtar Saeed University Act",
        "title_patterns": ["akhtar saeed university"],
        "keywords": [
            "akhtar saeed university", "university act", "higher education charter"
        ],
    },
    {
        "name": "National Assembly of Pakistan",
        "title_patterns": ["national assembly of pakistan"],
        "keywords": [
            "national assembly", "parliament", "speaker national assembly", "legislative procedure",
            "قومی اسمبلی", "پارلیمنٹ"
        ],
    },
    {
        "name": "Constitution (Twenty-sixth Amendment) Act",
        "title_patterns": ["twenty-sixth amendment", "26th amendment", "twenty sixth amendment"],
        "keywords": [
            "26th amendment", "twenty-sixth amendment", "constitutional benches", "article 175a", "article 191a",
            "article 202a", "article 9a", "chief justice appointment", "clean environment", "چھبیسویں ترمیم",
            "آئینی بینچ", "ماحولیات", "آئینی ترامیم"
        ],
    },
    {
        "name": "Supreme Court (Practice and Procedure) Act",
        "title_patterns": ["practice and procedure", "supreme court practice and procedure"],
        "keywords": [
            "practice and procedure", "supreme court practice and procedure act", "bench constitution",
            "appeal against 184(3)", "suo motu appeal", "30 days appeal", "سپریم کورٹ پریکٹس اینڈ پروسیجر",
            "آرٹیکل 184(3) کے خلاف اپیل", "بینچ کی تشکیل"
        ],
    },
    {
        "name": "Establishment of Special Court (Overseas Pakistanis Property) Act",
        "title_patterns": ["overseas pakistanis property", "special court overseas"],
        "keywords": [
            "overseas pakistanis", "overseas property", "special court overseas pakistanis property act 2024",
            "act xxviii of 2024", "overseas property dispute", "اوورسیز پاکستانی", "اوورسیز جائیداد",
            "خصوصی عدالت اوورسیز", "بیرون ملک پاکستانیوں کی جائیداد"
        ],
    },
    {
        "name": "Elections Act, 2017 & Election Rules",
        "title_patterns": ["elections act", "election rules", "election commission of pakistan"],
        "keywords": [
            "election act", "elections act 2017", "ecp", "election commission of pakistan", "nomination papers",
            "disqualification", "election petition", "election tribunal", "delimitation", "form 31", "rule 90",
            "reserved seats", "انتخابات ایکٹ", "الیکشن کمیشن", "انتخابی عذرداری", "مخصوص نشستیں", "انتخابی قوانین"
        ],
    },
    {
        "name": "Competition Act, 2010",
        "title_patterns": ["competition act", "competition commission"],
        "keywords": [
            "competition act", "competition commission of pakistan", "ccp", "abuse of dominant position",
            "prohibited agreement", "cartel", "deceptive marketing", "merger clearance",
            "موقر حیثیت کا ناجائز استعمال", "کمپٹیشن کمیشن", "اجارہ داری"
        ],
    },
    {
        "name": "State Bank of Pakistan Act & Banking Laws",
        "title_patterns": ["state bank of pakistan", "banking regulations", "sbp"],
        "keywords": [
            "state bank of pakistan", "sbp act", "banking companies ordinance", "prudential regulations",
            "monetary policy", "foreign exchange", "بینکنگ قوانین", "اسٹیٹ بینک", "فارن ایکسچینج"
        ],
    },
    {
        "name": "Anti-Money Laundering Regulations",
        "title_patterns": ["anti-money laundering", "aml/cft", "financial monitoring unit"],
        "keywords": [
            "money laundering", "aml", "cft", "anti money laundering act", "financial monitoring unit", "fmu",
            "suspicious transaction", "منی لانڈرنگ", "مشکوک مالیاتی لین دین"
        ],
    },
    {
        "name": "Pakistan Code",
        "title_patterns": ["pakistan code"],
        "keywords": [
            "pakistan code", "federal laws", "statutes of pakistan", "قوانین پاکستان"
        ],
    },
]


class Retriever:
    """Production-grade legal retriever with dense semantic + statutory hybrid search."""

    def __init__(
        self,
        index: Any | None = None,
        chunks: list[dict[str, Any]] | None = None,
        model: Any | None = None,
        manifest: dict[str, Any] | None = None,
        index_path: Path | str | None = None,
        chunks_path: Path | str | None = None,
        manifest_path: Path | str | None = None,
    ):
        idx_p = Path(index_path) if index_path else INDEX_PATH
        chk_p = Path(chunks_path) if chunks_path else CHUNKS_PATH
        man_p = Path(manifest_path) if manifest_path else MANIFEST_PATH

        if index is not None and chunks is not None:
            self.index = index
            self.chunks = chunks
        else:
            if not idx_p.exists() or not chk_p.exists():
                raise FileNotFoundError(
                    f"Knowledge base artifacts missing at {idx_p} or {chk_p}. "
                    "Run 'python -m app.embedder' or 'python -m app.loader' first."
                )

            self.index = faiss.read_index(str(idx_p))
            with open(chk_p, "rb") as f:
                self.chunks = pickle.load(f)

        # Validate vector and chunk counts
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"Knowledge base vector mismatch: FAISS index has {self.index.ntotal} vectors, "
                f"but chunks metadata has {len(self.chunks)} passages. "
                "The index and metadata are out of sync. Rebuild using 'python -m app.embedder'."
            )

        self.dimension = self.index.d
        self.metric = "cosine"

        # Manifest validation
        manifest_data = manifest
        if manifest_data is None and man_p.exists():
            try:
                with open(man_p, "r", encoding="utf-8") as mf:
                    manifest_data = json.load(mf)
            except Exception as exc:
                print(f"Warning: Could not read index manifest at {man_p}: {exc}")

        self.manifest = manifest_data
        if manifest_data:
            m_metric = str(manifest_data.get("metric", "cosine")).strip().casefold()
            m_dim = manifest_data.get("dimension")
            m_count = manifest_data.get("chunk_count") or manifest_data.get("total_chunks")
            m_model = manifest_data.get("embedding_model")

            if m_metric not in ("cosine", "inner_product", "ip"):
                raise ValueError(
                    f"Incompatible FAISS metric in manifest: '{m_metric}'. "
                    "The retriever requires cosine similarity (inner-product on L2-normalized embeddings). "
                    "Please rebuild the index using 'python -m app.embedder'."
                )
            self.metric = m_metric

            if m_dim is not None and int(m_dim) != self.dimension:
                raise ValueError(
                    f"Dimension mismatch: manifest declares dimension {m_dim}, "
                    f"but loaded FAISS index dimension is {self.dimension}."
                )

            if m_count is not None and int(m_count) != self.index.ntotal:
                raise ValueError(
                    f"Chunk count mismatch: manifest declares {m_count} chunks, "
                    f"but FAISS index contains {self.index.ntotal} vectors."
                )

            if m_model and model is None and m_model != EMBEDDING_MODEL:
                print(
                    f"Notice: manifest embedding model '{m_model}' differs from default "
                    f"config model '{EMBEDDING_MODEL}'."
                )

        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(EMBEDDING_MODEL)

    def _parse_query(self, query: str) -> dict[str, Any]:
        """Extract legal statutory entities, section numbers, orders, rules, and targeted statutes."""
        norm_query = query.translate(URDU_DIGITS_MAP)
        q_lower = norm_query.lower()

        # Extract explicit section numbers (e.g. section 302, دفعہ 489-F, article 199)
        raw_sections = SECTION_QUERY_REGEX.findall(q_lower)
        standalone_sections = STANDALONE_SECTION_REGEX.findall(q_lower)
        
        # Combine and deduplicate sections
        all_sections: list[str] = []
        for s in raw_sections + standalone_sections:
            clean_s = s.translate(URDU_DIGITS_MAP).strip().upper()
            if clean_s and clean_s not in all_sections:
                all_sections.append(clean_s)

        # Extract Order and Rule references (e.g. Order 39 Rule 2, آرڈر 39 رول 1)
        order_rules_raw = ORDER_RULE_REGEX.findall(q_lower)
        order_rules: list[tuple[str, str]] = []
        for ord_part, rule_part in order_rules_raw:
            ord_clean = ord_part.translate(URDU_DIGITS_MAP).strip()
            rule_clean = rule_part.translate(URDU_DIGITS_MAP).strip() if rule_part else ""
            order_rules.append((ord_clean, rule_clean))

        # Identify targeted statutes from STATUTE_CATALOG
        matched_statutes: list[str] = []
        matched_title_patterns: list[str] = []

        for entry in STATUTE_CATALOG:
            matched = False
            for kw in entry["keywords"]:
                if kw in q_lower:
                    matched = True
                    break
                if kw.isascii() and re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
                    matched = True
                    break
            if matched:
                matched_statutes.append(entry["name"])
                for pat in entry["title_patterns"]:
                    if pat not in matched_title_patterns:
                        matched_title_patterns.append(pat)

        return {
            "sections": all_sections,
            "order_rules": order_rules,
            "statutes": matched_statutes,
            "title_patterns": matched_title_patterns,
        }

    def retrieve(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """Retrieve relevant legal passages with semantic search and statutory boosting."""
        if not isinstance(query, str) or not query.strip():
            return []

        top_k = max(1, min(int(top_k), len(self.chunks)))
        parsed = self._parse_query(query)

        # 1. Semantic Embedding Search
        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)

        # Retrieve a broader candidate pool for re-ranking
        candidate_k = min(max(top_k * 5, 50), len(self.chunks))
        scores, indices = self.index.search(query_vec, candidate_k)

        candidates = []
        seen_indices = set()

        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                chunk = deepcopy(self.chunks[idx])
                chunk["semantic_score"] = float(score)
                chunk["_idx"] = int(idx)
                candidates.append(chunk)
                seen_indices.add(int(idx))

        # 2. Statutory Exact Match Injection
        # If user explicitly specified sections or Order/Rules, guarantee matching passages enter candidates
        if parsed["sections"] or parsed["order_rules"]:
            for idx, c in enumerate(self.chunks):
                if idx in seen_indices:
                    continue
                c_text = c.get("text", "").lower()
                c_sec = str(c.get("section") or "").lower()

                match = False

                # Section & Article matching
                for sec in parsed["sections"]:
                    sec_lower = sec.lower()
                    if f"section {sec_lower}" in c_sec or f"article {sec_lower}" in c_sec:
                        match = True
                        break
                    if re.search(r"\b" + re.escape(sec_lower) + r"\b", c_sec):
                        match = True
                        break
                    if re.search(r"\b(?:section|article|sec\.|art\.)\s+" + re.escape(sec_lower) + r"\b", c_text):
                        match = True
                        break

                # Order & Rule matching
                if not match and parsed["order_rules"]:
                    for ord_num, rule_num in parsed["order_rules"]:
                        ord_str = str(ord_num).lower()
                        ord_roman = INT_TO_ROMAN.get(int(ord_str), "").lower() if ord_str.isdigit() else ""
                        ord_hit = (
                            f"order {ord_str}" in c_sec
                            or f"order {ord_str}" in c_text
                            or (ord_roman and (f"order {ord_roman}" in c_sec or f"order {ord_roman}" in c_text))
                        )
                        if ord_hit:
                            if rule_num:
                                rule_str = str(rule_num).lower()
                                if f"rule {rule_str}" in c_sec or f"rule {rule_str}" in c_text:
                                    match = True
                                    break
                            else:
                                match = True
                                break

                if match:
                    chunk = deepcopy(c)
                    chunk["semantic_score"] = 0.50
                    chunk["_idx"] = idx
                    candidates.append(chunk)
                    seen_indices.add(idx)
                    if len(candidates) >= candidate_k + 35:
                        break

        # 3. Hybrid Re-Ranking & Scoring
        scored_candidates = []
        q_tokens = set(re.findall(r"\w+", query.lower()))

        for c in candidates:
            score = c.get("semantic_score", 0.0)
            c_text_lower = c.get("text", "").lower()
            c_title = c.get("document_title", "").lower()
            c_sec = str(c.get("section") or "").lower()

            # Boost exact section / article match
            for sec in parsed["sections"]:
                sec_lower = sec.lower()
                if sec_lower in c_sec:
                    score += 0.40
                    break
                elif re.search(r"\b(?:section|article|sec\.|art\.)\s+" + re.escape(sec_lower) + r"\b", c_text_lower):
                    score += 0.28
                    break

            # Boost exact Order / Rule match
            for ord_num, rule_num in parsed["order_rules"]:
                ord_str = str(ord_num).lower()
                ord_roman = INT_TO_ROMAN.get(int(ord_str), "").lower() if ord_str.isdigit() else ""
                ord_hit = (
                    f"order {ord_str}" in c_sec
                    or f"order {ord_str}" in c_text_lower
                    or (ord_roman and (f"order {ord_roman}" in c_sec or f"order {ord_roman}" in c_text_lower))
                )
                if ord_hit:
                    score += 0.30
                    if rule_num:
                        rule_str = str(rule_num).lower()
                        if f"rule {rule_str}" in c_sec or f"rule {rule_str}" in c_text_lower:
                            score += 0.20
                    break

            # Boost statute match from catalog
            for pattern in parsed["title_patterns"]:
                if pattern in c_title:
                    score += 0.30
                    break

            # Lexical token overlap bonus
            chunk_tokens = set(re.findall(r"\w+", c_text_lower))
            overlap = len(q_tokens & chunk_tokens) / max(1, len(q_tokens))
            score += overlap * 0.15

            c["hybrid_score"] = float(score)
            scored_candidates.append(c)

        # Sort by hybrid score descending
        scored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # 4. Diversity and Deduplication
        results = []
        seen_texts = set()

        for c in scored_candidates:
            # Dedup very similar chunks
            norm_snippet = " ".join(c.get("raw_text", c["text"])[:200].split())
            if norm_snippet in seen_texts:
                continue
            seen_texts.add(norm_snippet)
            results.append(c)
            if len(results) >= top_k:
                break

        return results
