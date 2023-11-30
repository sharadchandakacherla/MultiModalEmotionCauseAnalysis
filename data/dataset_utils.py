import collections

from transformers import AutoTokenizer


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

# def read_squad_examples(input_file, is_training, version_2_with_negative):
#     """Read a SQuAD json file into a list of SquadExample."""
#     # with open(input_file, "r", encoding='utf-8') as reader:
#     #     input_data = json.load(reader)["data"]
#     # input_data =[ {'title': 'Normans', 'paragraphs': [{'qas': [{'question': 'In what country is Normandy located?', 'id': '56ddde6b9a695914005b9628', 'answers': [{'text': 'France', 'answer_start': 159}, {'text': 'France', 'answer_start': 159}, {'text': 'France', 'answer_start': 159}, {'text': 'France', 'answer_start': 159}], 'is_impossible': False}, {'question': 'When were the Normans in Normandy?', 'id': '56ddde6b9a695914005b9629', 'answers': [{'text': '10th and 11th centuries', 'answer_start': 94}, {'text': 'in the 10th and 11th centuries', 'answer_start': 87}, {'text': '10th and 11th centuries', 'answer_start': 94}, {'text': '10th and 11th centuries', 'answer_start': 94}], 'is_impossible': False}, {'question': 'From which countries did the Norse originate?', 'id': '56ddde6b9a695914005b962a', 'answers': [{'text': 'Denmark, Iceland and Norway', 'answer_start': 256}, {'text': 'Denmark, Iceland and Norway', 'answer_start': 256}, {'text': 'Denmark, Iceland and Norway', 'answer_start': 256}, {'text': 'Denmark, Iceland and Norway', 'answer_start': 256}], 'is_impossible': False}, {'question': 'Who was the Norse leader?', 'id': '56ddde6b9a695914005b962b', 'answers': [{'text': 'Rollo', 'answer_start': 308}, {'text': 'Rollo', 'answer_start': 308}, {'text': 'Rollo', 'answer_start': 308}, {'text': 'Rollo', 'answer_start': 308}], 'is_impossible': False}, {'question': 'What century did the Normans first gain their separate identity?', 'id': '56ddde6b9a695914005b962c', 'answers': [{'text': '10th century', 'answer_start': 671}, {'text': 'the first half of the 10th century', 'answer_start': 649}, {'text': '10th', 'answer_start': 671}, {'text': '10th', 'answer_start': 671}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Normans', 'answer_start': 4}], 'question': "Who gave their name to Normandy in the 1000's and 1100's", 'id': '5ad39d53604f3c001a3fe8d1', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Normandy', 'answer_start': 137}], 'question': 'What is France a region of?', 'id': '5ad39d53604f3c001a3fe8d2', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Rollo', 'answer_start': 308}], 'question': 'Who did King Charles III swear fealty to?', 'id': '5ad39d53604f3c001a3fe8d3', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': '10th century', 'answer_start': 671}], 'question': 'When did the Frankish identity emerge?', 'id': '5ad39d53604f3c001a3fe8d4', 'answers': [], 'is_impossible': True}], 'context': 'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.'}, {'qas': [{'question': 'Who was the duke in the battle of Hastings?', 'id': '56dddf4066d3e219004dad5f', 'answers': [{'text': 'William the Conqueror', 'answer_start': 1022}, {'text': 'William the Conqueror', 'answer_start': 1022}, {'text': 'William the Conqueror', 'answer_start': 1022}], 'is_impossible': False}, {'question': 'Who ruled the duchy of Normandy', 'id': '56dddf4066d3e219004dad60', 'answers': [{'text': 'Richard I', 'answer_start': 573}, {'text': 'Richard I', 'answer_start': 573}, {'text': 'Richard I', 'answer_start': 573}], 'is_impossible': False}, {'question': 'What religion were the Normans', 'id': '56dddf4066d3e219004dad61', 'answers': [{'text': 'Catholic', 'answer_start': 230}, {'text': 'Catholic orthodoxy', 'answer_start': 230}, {'text': 'Catholic', 'answer_start': 230}], 'is_impossible': False}, {'plausible_answers': [{'text': 'political, cultural and military', 'answer_start': 31}], 'question': 'What type of major impact did the Norman dynasty have on modern Europe?', 'id': '5ad3a266604f3c001a3fea27', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Normans', 'answer_start': 118}], 'question': 'Who was famed for their Christian spirit?', 'id': '5ad3a266604f3c001a3fea28', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Normans', 'answer_start': 118}], 'question': 'Who assimilted the Roman language?', 'id': '5ad3a266604f3c001a3fea29', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Richard I', 'answer_start': 573}], 'question': 'Who ruled the country of Normandy?', 'id': '5ad3a266604f3c001a3fea2a', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Antioch', 'answer_start': 1295}], 'question': 'What principality did William the conquerer found?', 'id': '5ad3a266604f3c001a3fea2b', 'answers': [], 'is_impossible': True}], 'context': 'The Norman dynasty had a major political, cultural and military impact on medieval Europe and even the Near East. The Normans were famed for their martial spirit and eventually for their Christian piety, becoming exponents of the Catholic orthodoxy into which they assimilated. They adopted the Gallo-Romance language of the Frankish land they settled, their dialect becoming known as Norman, Normaund or Norman French, an important literary language. The Duchy of Normandy, which they formed by treaty with the French crown, was a great fief of medieval France, and under Richard I of Normandy was forged into a cohesive and formidable principality in feudal tenure. The Normans are noted both for their culture, such as their unique Romanesque architecture and musical traditions, and for their significant military accomplishments and innovations. Norman adventurers founded the Kingdom of Sicily under Roger II after conquering southern Italy on the Saracens and Byzantines, and an expedition on behalf of their duke, William the Conqueror, led to the Norman conquest of England at the Battle of Hastings in 1066. Norman cultural and military influence spread from these new European centres to the Crusader states of the Near East, where their prince Bohemond I founded the Principality of Antioch in the Levant, to Scotland and Wales in Great Britain, to Ireland, and to the coasts of north Africa and the Canary Islands.'}, {'qas': [{'question': 'What is the original meaning of the word Norman?', 'id': '56dde0379a695914005b9636', 'answers': [{'text': 'Viking', 'answer_start': 341}, {'text': 'Norseman, Viking', 'answer_start': 331}, {'text': 'Norseman, Viking', 'answer_start': 331}], 'is_impossible': False}, {'question': 'When was the Latin version of the word Norman first recorded?', 'id': '56dde0379a695914005b9637', 'answers': [{'text': '9th century', 'answer_start': 309}, {'text': '9th century', 'answer_start': 309}, {'text': '9th century', 'answer_start': 309}], 'is_impossible': False}, {'plausible_answers': [{'text': '"Normans"', 'answer_start': 17}], 'question': 'What name comes from the English words Normans/Normanz?', 'id': '5ad3ab70604f3c001a3feb89', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': '9th century', 'answer_start': 309}], 'question': 'When was the French version of the word Norman first recorded?', 'id': '5ad3ab70604f3c001a3feb8a', 'answers': [], 'is_impossible': True}], 'context': 'The English name "Normans" comes from the French words Normans/Normanz, plural of Normant, modern French normand, which is itself borrowed from Old Low Franconian Nortmann "Northman" or directly from Old Norse Norðmaðr, Latinized variously as Nortmannus, Normannus, or Nordmannus (recorded in Medieval Latin, 9th century) to mean "Norseman, Viking".'}, {'qas': [{'question': 'When was the Duchy of Normandy founded?', 'id': '56dde0ba66d3e219004dad75', 'answers': [{'text': '911', 'answer_start': 244}, {'text': '911', 'answer_start': 244}, {'text': '911', 'answer_start': 244}], 'is_impossible': False}, {'question': 'Who did Rollo sign the treaty of Saint-Clair-sur-Epte with?', 'id': '56dde0ba66d3e219004dad76', 'answers': [{'text': 'King Charles III', 'answer_start': 324}, {'text': 'King Charles III', 'answer_start': 324}, {'text': 'King Charles III', 'answer_start': 324}], 'is_impossible': False}, {'question': 'What river originally bounded the Duchy', 'id': '56dde0ba66d3e219004dad77', 'answers': [{'text': 'Seine', 'answer_start': 711}, {'text': 'Epte', 'answer_start': 524}, {'text': 'Seine', 'answer_start': 711}], 'is_impossible': False}, {'plausible_answers': [{'text': '10th century', 'answer_start': 21}], 'question': 'when did Nors encampments ivolve into destructive incursions?', 'id': '5ad3ad61604f3c001a3fec0d', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'treaty of Saint-Clair-sur-Epte', 'answer_start': 285}], 'question': 'What treaty was established in the 9th century?', 'id': '5ad3ad61604f3c001a3fec0e', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Rollo', 'answer_start': 384}], 'question': 'Who established a treaty with King Charles the third of France?', 'id': '5ad3ad61604f3c001a3fec0f', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'further Viking incursions.', 'answer_start': 593}], 'question': 'What did the French promises to protect Rollo and his men from?', 'id': '5ad3ad61604f3c001a3fec10', 'answers': [], 'is_impossible': True}], 'context': 'In the course of the 10th century, the initially destructive incursions of Norse war bands into the rivers of France evolved into more permanent encampments that included local women and personal property. The Duchy of Normandy, which began in 911 as a fiefdom, was established by the treaty of Saint-Clair-sur-Epte between King Charles III of West Francia and the famed Viking ruler Rollo, and was situated in the former Frankish kingdom of Neustria. The treaty offered Rollo and his men the French lands between the river Epte and the Atlantic coast in exchange for their protection against further Viking incursions. The area corresponded to the northern part of present-day Upper Normandy down to the river Seine, but the Duchy would eventually extend west beyond the Seine. The territory was roughly equivalent to the old province of Rouen, and reproduced the Roman administrative structure of Gallia Lugdunensis II (part of the former Gallia Lugdunensis).'}, {'qas': [{'question': 'Who upon arriving gave the original viking settlers a common identity?', 'id': '56dde1d966d3e219004dad8d', 'answers': [{'text': 'Rollo', 'answer_start': 7}, {'text': 'Rollo', 'answer_start': 7}, {'text': 'Rollo', 'answer_start': 7}], 'is_impossible': False}, {'plausible_answers': [{'text': '880s', 'answer_start': 174}], 'question': 'When did Rollo begin to arrive in Normandy?', 'id': '5ad3ae14604f3c001a3fec39', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Danes, Norwegians, Norse–Gaels, Orkney Vikings, possibly Swedes, and Anglo-Danes', 'answer_start': 547}], 'question': 'What Viking groups were conquered by Rollo?', 'id': '5ad3ae14604f3c001a3fec3a', 'answers': [], 'is_impossible': True}], 'context': 'Before Rollo\'s arrival, its populations did not differ from Picardy or the Île-de-France, which were considered "Frankish". Earlier Viking settlers had begun arriving in the 880s, but were divided between colonies in the east (Roumois and Pays de Caux) around the low Seine valley and in the west in the Cotentin Peninsula, and were separated by traditional pagii, where the population remained about the same with almost no foreign settlers. Rollo\'s contingents who raided and ultimately settled Normandy and parts of the Atlantic coast included Danes, Norwegians, Norse–Gaels, Orkney Vikings, possibly Swedes, and Anglo-Danes from the English Danelaw under Norse control.'}, {'qas': [{'question': 'What was the Norman religion?', 'id': '56dde27d9a695914005b9651', 'answers': [{'text': 'Catholicism', 'answer_start': 121}, {'text': 'Catholicism', 'answer_start': 121}, {'text': 'Catholicism', 'answer_start': 121}], 'is_impossible': False}, {'question': 'What part of France were the Normans located?', 'id': '56dde27d9a695914005b9652', 'answers': [{'text': 'north', 'answer_start': 327}, {'text': 'the north', 'answer_start': 323}, {'text': 'north', 'answer_start': 327}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Catholicism', 'answer_start': 121}], 'question': 'What was replace with the Norse religion?', 'id': '5ad3af11604f3c001a3fec63', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Frankish heritage', 'answer_start': 224}], 'question': 'What did maternal Old Norse traditions merge with?', 'id': '5ad3af11604f3c001a3fec64', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Old Norse', 'answer_start': 97}], 'question': 'What language replaced the Gallo-Romance language?', 'id': '5ad3af11604f3c001a3fec65', 'answers': [], 'is_impossible': True}], 'context': 'The descendants of Rollo\'s Vikings and their Frankish wives would replace the Norse religion and Old Norse language with Catholicism (Christianity) and the Gallo-Romance language of the local people, blending their maternal Frankish heritage with Old Norse traditions and customs to synthesize a unique "Norman" culture in the north of France. The Norman language was forged by the adoption of the indigenous langue d\'oïl branch of Romance by a Norse-speaking ruling class, and it developed into the regional language that survives today.'}, {'qas': [{'question': "What was one of the Norman's major exports?", 'id': '56dde2fa66d3e219004dad9b', 'answers': [{'text': 'fighting horsemen', 'answer_start': 428}, {'text': 'fighting horsemen', 'answer_start': 428}, {'text': 'fighting horsemen', 'answer_start': 428}], 'is_impossible': False}, {'plausible_answers': [{'text': 'France', 'answer_start': 75}], 'question': 'Who adopted the fuedel doctrines of the Normans?', 'id': '5ad3c626604f3c001a3ff011', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'fighting horsemen', 'answer_start': 428}], 'question': "What was one of the Norman's major imports?", 'id': '5ad3c626604f3c001a3ff012', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Italy, France and England', 'answer_start': 490}], 'question': "Who's arristocracy eventually served as avid Crusaders?", 'id': '5ad3c626604f3c001a3ff013', 'answers': [], 'is_impossible': True}], 'context': 'The Normans thereafter adopted the growing feudal doctrines of the rest of France, and worked them into a functional hierarchical system in both Normandy and in England. The new Norman rulers were culturally and ethnically distinct from the old French aristocracy, most of whom traced their lineage to Franks of the Carolingian dynasty. Most Norman knights remained poor and land-hungry, and by 1066 Normandy had been exporting fighting horsemen for more than a generation. Many Normans of Italy, France and England eventually served as avid Crusaders under the Italo-Norman prince Bohemund I and the Anglo-Norman king Richard the Lion-Heart.'}, {'qas': [{'question': "Who was the Normans' main enemy in Italy, the Byzantine Empire and Armenia?", 'id': '56de0f6a4396321400ee257f', 'answers': [{'text': 'Seljuk Turks', 'answer_start': 161}, {'text': 'the Pechenegs, the Bulgars, and especially the Seljuk Turks', 'answer_start': 114}, {'text': 'the Seljuk Turks', 'answer_start': 157}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Normans', 'answer_start': 15}], 'question': 'Who entered Italy soon after the Byzantine Empire?', 'id': '5ad3dbc6604f3c001a3ff3e9', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Pechenegs, the Bulgars, and especially the Seljuk Turks', 'answer_start': 118}], 'question': 'Who did the Normans fight in Italy?', 'id': '5ad3dbc6604f3c001a3ff3ea', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Lombards', 'answer_start': 244}], 'question': 'Who did the Normans encourage to come to the south?', 'id': '5ad3dbc6604f3c001a3ff3eb', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'the Sicilian campaign of George Maniaces', 'answer_start': 404}], 'question': 'During what campaign did the Vargian and Lombard fight?', 'id': '5ad3dbc6604f3c001a3ff3ec', 'answers': [], 'is_impossible': True}], 'context': 'Soon after the Normans began to enter Italy, they entered the Byzantine Empire and then Armenia, fighting against the Pechenegs, the Bulgars, and especially the Seljuk Turks. Norman mercenaries were first encouraged to come to the south by the Lombards to act against the Byzantines, but they soon fought in Byzantine service in Sicily. They were prominent alongside Varangian and Lombard contingents in the Sicilian campaign of George Maniaces in 1038–40. There is debate whether the Normans in Greek service actually were from Norman Italy, and it now seems likely only a few came from there. It is also unknown how many of the "Franks", as the Byzantines called them, were Normans and not other Frenchmen.'}, {'qas': [{'question': 'When did Herve serve as a Byzantine general?', 'id': '56de0ffd4396321400ee258d', 'answers': [{'text': '1050s', 'answer_start': 85}, {'text': 'in the 1050s', 'answer_start': 78}, {'text': 'in the 1050s', 'answer_start': 78}], 'is_impossible': False}, {'question': 'When did Robert Crispin go up against the Turks?', 'id': '56de0ffd4396321400ee258e', 'answers': [{'text': '1060s', 'answer_start': 292}, {'text': 'In the 1060s', 'answer_start': 285}, {'text': 'In the 1060s', 'answer_start': 285}], 'is_impossible': False}, {'question': "Who ruined Roussel de Bailleul's plans for an independent state?", 'id': '56de0ffd4396321400ee258f', 'answers': [{'text': 'Alexius Komnenos', 'answer_start': 522}, {'text': 'Alexius Komnenos', 'answer_start': 522}, {'text': 'Alexius Komnenos', 'answer_start': 522}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Hervé', 'answer_start': 72}], 'question': 'Who was the first Byzantine mercenary to serve with the Normans?', 'id': '5ad3de8b604f3c001a3ff467', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': '1050s', 'answer_start': 85}], 'question': 'When did Herve serve as a Norman general?', 'id': '5ad3de8b604f3c001a3ff468', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Roussel de Bailleul', 'answer_start': 359}], 'question': 'Who ruined Alexius Komnenos plans for an independent state?', 'id': '5ad3de8b604f3c001a3ff469', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': '1060s', 'answer_start': 292}], 'question': 'When did Herve go up against the Turks?', 'id': '5ad3de8b604f3c001a3ff46a', 'answers': [], 'is_impossible': True}], 'context': 'One of the first Norman mercenaries to serve as a Byzantine general was Hervé in the 1050s. By then however, there were already Norman mercenaries serving as far away as Trebizond and Georgia. They were based at Malatya and Edessa, under the Byzantine duke of Antioch, Isaac Komnenos. In the 1060s, Robert Crispin led the Normans of Edessa against the Turks. Roussel de Bailleul even tried to carve out an independent state in Asia Minor with support from the local population, but he was stopped by the Byzantine general Alexius Komnenos.'}, {'qas': [{'question': 'What was the name of the Norman castle?', 'id': '56de10b44396321400ee2593', 'answers': [{'text': 'Afranji', 'answer_start': 539}, {'text': 'Afranji', 'answer_start': 539}, {'text': 'Afranji', 'answer_start': 539}], 'is_impossible': False}, {'question': 'Who was the leader when the Franks entered the Euphrates valley?', 'id': '56de10b44396321400ee2594', 'answers': [{'text': 'Oursel', 'answer_start': 256}, {'text': 'Oursel', 'answer_start': 256}, {'text': 'Oursel', 'answer_start': 256}], 'is_impossible': False}, {'question': 'Who did the Normans team up with in Anatolia?', 'id': '56de10b44396321400ee2595', 'answers': [{'text': 'Turkish forces', 'answer_start': 20}, {'text': 'Turkish forces', 'answer_start': 20}, {'text': 'Turkish forces', 'answer_start': 20}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Turkish', 'answer_start': 20}], 'question': 'Who joined Norman forces in the destruction of the Armenians?', 'id': '5ad3e96b604f3c001a3ff689', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'the Armenian state', 'answer_start': 171}], 'question': 'Who did the Turks take up service with?', 'id': '5ad3e96b604f3c001a3ff68a', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Oursel', 'answer_start': 256}], 'question': 'What Frank led Norman forces?', 'id': '5ad3e96b604f3c001a3ff68b', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'the upper Euphrates valley in northern Syria', 'answer_start': 292}], 'question': 'Where did Oursel lead the Franks?', 'id': '5ad3e96b604f3c001a3ff68c', 'answers': [], 'is_impossible': True}], 'context': 'Some Normans joined Turkish forces to aid in the destruction of the Armenians vassal-states of Sassoun and Taron in far eastern Anatolia. Later, many took up service with the Armenian state further south in Cilicia and the Taurus Mountains. A Norman named Oursel led a force of "Franks" into the upper Euphrates valley in northern Syria. From 1073 to 1074, 8,000 of the 20,000 troops of the Armenian general Philaretus Brachamius were Normans—formerly of Oursel—led by Raimbaud. They even lent their ethnicity to the name of their castle: Afranji, meaning "Franks." The known trade between Amalfi and Antioch and between Bari and Tarsus may be related to the presence of Italo-Normans in those cities while Amalfi and Bari were under Norman rule in Italy.'}, {'qas': [{'question': 'What were the origins of the Raouliii family?', 'id': '56de11154396321400ee25aa', 'answers': [{'text': 'Norman mercenary', 'answer_start': 45}, {'text': 'an Italo-Norman named Raoul', 'answer_start': 217}, {'text': 'descended from an Italo-Norman named Raoul', 'answer_start': 202}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Byzantine Greece', 'answer_start': 20}], 'question': 'Where were several Norman mercenary familes originate from?', 'id': '5ad3ea79604f3c001a3ff6e9', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'George Maniaces', 'answer_start': 402}], 'question': 'Who did the Normans serve under in the 10th century?', 'id': '5ad3ea79604f3c001a3ff6ea', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Sicilian expedition', 'answer_start': 425}], 'question': 'What expedition did George Maniaces lead in the 10th century?', 'id': '5ad3ea79604f3c001a3ff6eb', 'answers': [], 'is_impossible': True}], 'context': "Several families of Byzantine Greece were of Norman mercenary origin during the period of the Comnenian Restoration, when Byzantine emperors were seeking out western European warriors. The Raoulii were descended from an Italo-Norman named Raoul, the Petraliphae were descended from a Pierre d'Aulps, and that group of Albanian clans known as the Maniakates were descended from Normans who served under George Maniaces in the Sicilian expedition of 1038."}, {'qas': [{'question': 'What was the name of the count of Apulia ', 'id': '56de148dcffd8e1900b4b5bc', 'answers': [{'text': 'Robert Guiscard', 'answer_start': 0}, {'text': 'Robert Guiscard', 'answer_start': 0}, {'text': 'Robert Guiscard', 'answer_start': 0}], 'is_impossible': False}, {'question': 'When did Dyrrachium  fall to the Normans?', 'id': '56de148dcffd8e1900b4b5bd', 'answers': [{'text': '1082', 'answer_start': 1315}, {'text': 'February 1082', 'answer_start': 1306}, {'text': 'February 1082', 'answer_start': 1306}], 'is_impossible': False}, {'question': "How many men were in Robert's army?", 'id': '56de148dcffd8e1900b4b5be', 'answers': [{'text': '30,000', 'answer_start': 492}, {'text': '30,000', 'answer_start': 492}, {'text': '30,000', 'answer_start': 492}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Robert Guiscard', 'answer_start': 0}], 'question': 'Who ultimatly drove the Byzantines out of Europe?', 'id': '5ad3ed26604f3c001a3ff799', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'pope Gregory VII', 'answer_start': 225}], 'question': 'What pope opposed Roberts campaign?', 'id': '5ad3ed26604f3c001a3ff79a', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Dyrrachium', 'answer_start': 1326}], 'question': 'What fell to the Normans in the 10th century?', 'id': '5ad3ed26604f3c001a3ff79b', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': '30,000', 'answer_start': 492}], 'question': 'How many men did Roberts army face?', 'id': '5ad3ed26604f3c001a3ff79c', 'answers': [], 'is_impossible': True}], 'context': "Robert Guiscard, an other Norman adventurer previously elevated to the dignity of count of Apulia as the result of his military successes, ultimately drove the Byzantines out of southern Italy. Having obtained the consent of pope Gregory VII and acting as his vassal, Robert continued his campaign conquering the Balkan peninsula as a foothold for western feudal lords and the Catholic Church. After allying himself with Croatia and the Catholic cities of Dalmatia, in 1081 he led an army of 30,000 men in 300 ships landing on the southern shores of Albania, capturing Valona, Kanina, Jericho (Orikumi), and reaching Butrint after numerous pillages. They joined the fleet that had previously conquered Corfu and attacked Dyrrachium from land and sea, devastating everything along the way. Under these harsh circumstances, the locals accepted the call of emperor Alexius I Comnenus to join forces with the Byzantines against the Normans. The Albanian forces could not take part in the ensuing battle because it had started before their arrival. Immediately before the battle, the Venetian fleet had secured a victory in the coast surrounding the city. Forced to retreat, Alexius ceded the command to a high Albanian official named Comiscortes in the service of Byzantium. The city's garrison resisted until February 1082, when Dyrrachium was betrayed to the Normans by the Venetian and Amalfitan merchants who had settled there. The Normans were now free to penetrate into the hinterland; they took Ioannina and some minor cities in southwestern Macedonia and Thessaly before appearing at the gates of Thessalonica. Dissension among the high ranks coerced the Normans to retreat to Italy. They lost Dyrrachium, Valona, and Butrint in 1085, after the death of Robert."}, {'qas': [{'question': 'Where did the Normans and Byzantines sign the peace treaty?', 'id': '56de15104396321400ee25b7', 'answers': [{'text': 'Deabolis', 'answer_start': 302}, {'text': 'Deabolis', 'answer_start': 718}, {'text': 'Deabolis', 'answer_start': 718}], 'is_impossible': False}, {'question': "Who was Robert's son?", 'id': '56de15104396321400ee25b8', 'answers': [{'text': 'Bohemond', 'answer_start': 79}, {'text': 'Bohemond', 'answer_start': 79}, {'text': 'Bohemond', 'answer_start': 79}], 'is_impossible': False}, {'question': 'What river was Petrela located by?', 'id': '56de15104396321400ee25b9', 'answers': [{'text': 'Deabolis', 'answer_start': 302}, {'text': 'the river Deabolis', 'answer_start': 292}, {'text': 'Deabolis', 'answer_start': 302}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Dyrrachium', 'answer_start': 133}], 'question': 'Who did the Normans besiege in the 11th century?', 'id': '5ad3ee2d604f3c001a3ff7e1', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Normans', 'answer_start': 50}], 'question': 'Who did Robert lead agains Dyrrachium in 1107?', 'id': '5ad3ee2d604f3c001a3ff7e2', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Robert', 'answer_start': 89}], 'question': "Who was Bohemond's son?", 'id': '5ad3ee2d604f3c001a3ff7e3', 'answers': [], 'is_impossible': True}], 'context': "A few years after the First Crusade, in 1107, the Normans under the command of Bohemond, Robert's son, landed in Valona and besieged Dyrrachium using the most sophisticated military equipment of the time, but to no avail. Meanwhile, they occupied Petrela, the citadel of Mili at the banks of the river Deabolis, Gllavenica (Ballsh), Kanina and Jericho. This time, the Albanians sided with the Normans, dissatisfied by the heavy taxes the Byzantines had imposed upon them. With their help, the Normans secured the Arbanon passes and opened their way to Dibra. The lack of supplies, disease and Byzantine resistance forced Bohemond to retreat from his campaign and sign a peace treaty with the Byzantines in the city of Deabolis."}, {'qas': [{'question': 'When did the Normans attack Dyrrachium?', 'id': '56de1563cffd8e1900b4b5c2', 'answers': [{'text': '1185', 'answer_start': 86}, {'text': 'in 1185', 'answer_start': 83}, {'text': '1185', 'answer_start': 86}], 'is_impossible': False}, {'question': 'What was the naval base called?', 'id': '56de1563cffd8e1900b4b5c3', 'answers': [{'text': 'Dyrrachium', 'answer_start': 125}, {'text': 'Dyrrachium', 'answer_start': 205}, {'text': 'Dyrrachium', 'answer_start': 205}], 'is_impossible': False}, {'question': 'Where was Dyrrachium located?', 'id': '56de1563cffd8e1900b4b5c4', 'answers': [{'text': 'the Adriatic', 'answer_start': 257}, {'text': 'the Adriatic', 'answer_start': 257}, {'text': 'Adriatic', 'answer_start': 261}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Norman army', 'answer_start': 105}], 'question': 'Who attacked Dyrrachium in the 11th century?', 'id': '5ad3f028604f3c001a3ff823', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'high Byzantine officials', 'answer_start': 162}], 'question': 'Who betrayed the Normans?', 'id': '5ad3f028604f3c001a3ff824', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Dyrrachium', 'answer_start': 205}], 'question': 'What naval base fell to the Normans?', 'id': '5ad3f028604f3c001a3ff825', 'answers': [], 'is_impossible': True}], 'context': 'The further decline of Byzantine state-of-affairs paved the road to a third attack in 1185, when a large Norman army invaded Dyrrachium, owing to the betrayal of high Byzantine officials. Some time later, Dyrrachium—one of the most important naval bases of the Adriatic—fell again to Byzantine hands.'}, {'qas': [{'question': 'Who did Emma Marry?', 'id': '56de15dbcffd8e1900b4b5c8', 'answers': [{'text': 'King Ethelred II', 'answer_start': 360}, {'text': 'Ethelred II', 'answer_start': 365}, {'text': 'King Ethelred II', 'answer_start': 360}], 'is_impossible': False}, {'question': "Who was Emma's brother?", 'id': '56de15dbcffd8e1900b4b5c9', 'answers': [{'text': 'Duke Richard II', 'answer_start': 327}, {'text': 'Duke Richard II', 'answer_start': 327}, {'text': 'Duke Richard II', 'answer_start': 327}], 'is_impossible': False}, {'question': 'To where did Ethelred flee?', 'id': '56de15dbcffd8e1900b4b5ca', 'answers': [{'text': 'Normandy', 'answer_start': 423}, {'text': 'Normandy', 'answer_start': 423}, {'text': 'Normandy', 'answer_start': 423}], 'is_impossible': False}, {'question': 'Who kicked Ethelred out?', 'id': '56de15dbcffd8e1900b4b5cb', 'answers': [{'text': 'Sweyn Forkbeard', 'answer_start': 480}, {'text': 'Sweyn Forkbeard', 'answer_start': 480}, {'text': 'Sweyn Forkbeard', 'answer_start': 480}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Emma', 'answer_start': 562}], 'question': 'Who married Cnut the Great?', 'id': '5ad3f187604f3c001a3ff86f', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': '1013', 'answer_start': 435}], 'question': 'When did Richard II flee to Normandy?', 'id': '5ad3f187604f3c001a3ff870', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Viking', 'answer_start': 90}], 'question': "Who's major ports were controlled by the English?", 'id': '5ad3f187604f3c001a3ff871', 'answers': [], 'is_impossible': True}], 'context': "The Normans were in contact with England from an early date. Not only were their original Viking brethren still ravaging the English coasts, they occupied most of the important ports opposite England across the English Channel. This relationship eventually produced closer ties of blood through the marriage of Emma, sister of Duke Richard II of Normandy, and King Ethelred II of England. Because of this, Ethelred fled to Normandy in 1013, when he was forced from his kingdom by Sweyn Forkbeard. His stay in Normandy (until 1016) influenced him and his sons by Emma, who stayed in Normandy after Cnut the Great's conquest of the isle."}, {'qas': [{'question': "Who was Edward the Confessor's half-brother?", 'id': '56de1645cffd8e1900b4b5d0', 'answers': [{'text': 'Harthacnut', 'answer_start': 115}, {'text': 'Harthacnut', 'answer_start': 115}, {'text': 'Harthacnut', 'answer_start': 115}], 'is_impossible': False}, {'question': 'When did Edward return?', 'id': '56de1645cffd8e1900b4b5d1', 'answers': [{'text': '1041', 'answer_start': 71}, {'text': 'in 1041', 'answer_start': 68}, {'text': '1041', 'answer_start': 71}], 'is_impossible': False}, {'question': 'Who did Edward make archbishop of Canterbury?', 'id': '56de1645cffd8e1900b4b5d2', 'answers': [{'text': 'Robert of Jumièges', 'answer_start': 382}, {'text': 'Robert of Jumièges', 'answer_start': 382}, {'text': 'Robert of Jumièges', 'answer_start': 382}], 'is_impossible': False}, {'plausible_answers': [{'text': '1041', 'answer_start': 71}], 'question': "When did Edward the Confessor's son return from his fathers refuge?", 'id': '5ad3f350604f3c001a3ff8ef', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'English cavalry force', 'answer_start': 253}], 'question': 'What kind of force did Harthacnut establish?', 'id': '5ad3f350604f3c001a3ff8f0', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Edward', 'answer_start': 361}], 'question': 'Who made Robert of Jumieges earl of Hereford?', 'id': '5ad3f350604f3c001a3ff8f1', 'answers': [], 'is_impossible': True}], 'context': "When finally Edward the Confessor returned from his father's refuge in 1041, at the invitation of his half-brother Harthacnut, he brought with him a Norman-educated mind. He also brought many Norman counsellors and fighters, some of whom established an English cavalry force. This concept never really took root, but it is a typical example of the attitudes of Edward. He appointed Robert of Jumièges archbishop of Canterbury and made Ralph the Timid earl of Hereford. He invited his brother-in-law Eustace II, Count of Boulogne to his court in 1051, an event which resulted in the greatest of early conflicts between Saxon and Norman and ultimately resulted in the exile of Earl Godwin of Wessex."}, {'qas': [{'question': 'Where did Harold II die?', 'id': '56de16ca4396321400ee25c5', 'answers': [{'text': 'Battle of Hastings', 'answer_start': 85}, {'text': 'the Battle of Hastings', 'answer_start': 81}, {'text': 'at the Battle of Hastings', 'answer_start': 78}], 'is_impossible': False}, {'question': 'Who killed Harold II? ', 'id': '56de16ca4396321400ee25c6', 'answers': [{'text': 'William II', 'answer_start': 14}, {'text': 'Duke William II', 'answer_start': 9}, {'text': 'Duke William II', 'answer_start': 9}], 'is_impossible': False}, {'question': 'When was the Battle of Hastings?', 'id': '56de16ca4396321400ee25c7', 'answers': [{'text': '1066', 'answer_start': 3}, {'text': 'In 1066', 'answer_start': 0}, {'text': '1066', 'answer_start': 3}], 'is_impossible': False}, {'question': 'Who was the ruling class ahead of the Normans?', 'id': '56de16ca4396321400ee25c8', 'answers': [{'text': 'Anglo-Saxons', 'answer_start': 161}, {'text': 'the Anglo-Saxons', 'answer_start': 157}, {'text': 'Anglo-Saxons', 'answer_start': 161}], 'is_impossible': False}, {'plausible_answers': [{'text': '1066,', 'answer_start': 3}], 'question': 'When did King Harold II conquer England?', 'id': '5ad3f4b1604f3c001a3ff951', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Battle of Hastings', 'answer_start': 85}], 'question': 'What battle took place in the 10th century?', 'id': '5ad3f4b1604f3c001a3ff952', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Anglo-Saxons', 'answer_start': 161}], 'question': 'Who replaced the Normans as the ruling class?', 'id': '5ad3f4b1604f3c001a3ff953', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Early Norman kings', 'answer_start': 317}], 'question': 'Who considered their land on the continent their most important holding?', 'id': '5ad3f4b1604f3c001a3ff954', 'answers': [], 'is_impossible': True}], 'context': 'In 1066, Duke William II of Normandy conquered England killing King Harold II at the Battle of Hastings. The invading Normans and their descendants replaced the Anglo-Saxons as the ruling class of England. The nobility of England were part of a single Normans culture and many had lands on both sides of the channel. Early Norman kings of England, as Dukes of Normandy, owed homage to the King of France for their land on the continent. They considered England to be their most important holding (it brought with it the title of King—an important status symbol).'}, {'qas': [{'question': "What was the Anglo-Norman language's final form?", 'id': '56de1728cffd8e1900b4b5d7', 'answers': [{'text': 'Modern English', 'answer_start': 629}, {'text': 'Modern English', 'answer_start': 629}, {'text': 'Modern English', 'answer_start': 629}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Norman aristocracy', 'answer_start': 130}], 'question': 'Who identified themselves as French during the Hundred Years War?', 'id': '5ad3f5b0604f3c001a3ff9ab', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Anglo-Saxon', 'answer_start': 382}], 'question': 'What was absorbed into the Anglo-Norman language?', 'id': '5ad3f5b0604f3c001a3ff9ac', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Geoffrey Chaucer', 'answer_start': 305}], 'question': 'Who made fun of the Latin language?', 'id': '5ad3f5b0604f3c001a3ff9ad', 'answers': [], 'is_impossible': True}], 'context': "Eventually, the Normans merged with the natives, combining languages and traditions. In the course of the Hundred Years' War, the Norman aristocracy often identified themselves as English. The Anglo-Norman language became distinct from the Latin language, something that was the subject of some humour by Geoffrey Chaucer. The Anglo-Norman language was eventually absorbed into the Anglo-Saxon language of their subjects (see Old English) and influenced it, helping (along with the Norse language of the earlier Anglo-Norse settlers and the Latin used by the church) in the development of Middle English. It in turn evolved into Modern English."}, {'qas': [{'question': "In what year did the Norman's invade at Bannow Bay?", 'id': '56de179dcffd8e1900b4b5da', 'answers': [{'text': '1169', 'answer_start': 101}, {'text': '1169', 'answer_start': 101}, {'text': '1169', 'answer_start': 101}], 'is_impossible': False}, {'question': 'What country did the Normans invade in 1169?', 'id': '56de179dcffd8e1900b4b5db', 'answers': [{'text': 'Ireland', 'answer_start': 379}, {'text': 'Ireland', 'answer_start': 379}, {'text': 'Ireland', 'answer_start': 379}], 'is_impossible': False}, {'question': 'What culture did the Normans combine with in Ireland?', 'id': '56de179dcffd8e1900b4b5dc', 'answers': [{'text': 'Irish', 'answer_start': 37}, {'text': 'Irish', 'answer_start': 220}, {'text': 'Irish', 'answer_start': 220}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Bannow Bay', 'answer_start': 87}], 'question': 'Where did the Normans invade in the 11th century?', 'id': '5ad3f6f5604f3c001a3ffa09', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'The Normans', 'answer_start': 0}], 'question': 'Who did the Irish culture have a profound effect on?', 'id': '5ad3f6f5604f3c001a3ffa0a', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Trim Castle and Dublin Castle', 'answer_start': 473}], 'question': 'What castles were built by the Irish?', 'id': '5ad3f6f5604f3c001a3ffa0b', 'answers': [], 'is_impossible': True}], 'context': 'The Normans had a profound effect on Irish culture and history after their invasion at Bannow Bay in 1169. Initially the Normans maintained a distinct culture and ethnicity. Yet, with time, they came to be subsumed into Irish culture to the point that it has been said that they became "more Irish than the Irish themselves." The Normans settled mostly in an area in the east of Ireland, later known as the Pale, and also built many fine castles and settlements, including Trim Castle and Dublin Castle. Both cultures intermixed, borrowing from each other\'s language, culture and outlook. Norman descendants today can be recognised by their surnames. Names such as French, (De) Roche, Devereux, D\'Arcy, Treacy and Lacy are particularly common in the southeast of Ireland, especially in the southern part of County Wexford where the first Norman settlements were established. Other Norman names such as Furlong predominate there. Another common Norman-Irish name was Morell (Murrell) derived from the French Norman name Morel. Other names beginning with Fitz (from the Norman for son) indicate Norman ancestry. These included Fitzgerald, FitzGibbons (Gibbons) dynasty, Fitzmaurice. Other families bearing such surnames as Barry (de Barra) and De Búrca (Burke) are also of Norman extraction.'}, {'qas': [{'question': "Who was Margaret's brother?", 'id': '56de17f9cffd8e1900b4b5e0', 'answers': [{'text': 'Edgar', 'answer_start': 75}, {'text': 'Edgar', 'answer_start': 157}, {'text': 'Edgar Atheling', 'answer_start': 75}], 'is_impossible': False}, {'question': "Who was Margaret's husband?", 'id': '56de17f9cffd8e1900b4b5e1', 'answers': [{'text': 'King Malcolm III of Scotland', 'answer_start': 120}, {'text': 'King Malcolm III', 'answer_start': 120}, {'text': 'King Malcolm III', 'answer_start': 120}], 'is_impossible': False}, {'question': 'When was Scotland invaded by William?', 'id': '56de17f9cffd8e1900b4b5e2', 'answers': [{'text': '1072', 'answer_start': 300}, {'text': '1072', 'answer_start': 300}, {'text': '1072', 'answer_start': 300}], 'is_impossible': False}, {'question': 'Who was the hostage?', 'id': '56de17f9cffd8e1900b4b5e3', 'answers': [{'text': 'Duncan', 'answer_start': 440}, {'text': 'Duncan', 'answer_start': 440}, {'text': 'Duncan', 'answer_start': 440}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Margaret', 'answer_start': 172}], 'question': 'Who did Edgar marry?', 'id': '5ad3f7ac604f3c001a3ffa3b', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'William', 'answer_start': 272}], 'question': 'Who invaded Scotland in the 10th century?', 'id': '5ad3f7ac604f3c001a3ffa3c', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Duncan', 'answer_start': 440}], 'question': 'Who did the Scotish king take hostage?', 'id': '5ad3f7ac604f3c001a3ffa3d', 'answers': [], 'is_impossible': True}], 'context': "One of the claimants of the English throne opposing William the Conqueror, Edgar Atheling, eventually fled to Scotland. King Malcolm III of Scotland married Edgar's sister Margaret, and came into opposition to William who had already disputed Scotland's southern borders. William invaded Scotland in 1072, riding as far as Abernethy where he met up with his fleet of ships. Malcolm submitted, paid homage to William and surrendered his son Duncan as a hostage, beginning a series of arguments as to whether the Scottish Crown owed allegiance to the King of England."}, {'qas': [{'question': 'Who did Alexander I marry?', 'id': '56de3cd0cffd8e1900b4b6be', 'answers': [{'text': 'Sybilla of Normandy', 'answer_start': 271}, {'text': 'Sybilla of Normandy', 'answer_start': 271}, {'text': 'Sybilla', 'answer_start': 271}], 'is_impossible': False}, {'question': 'What culture\'s arrival in Scotland is know as the "Davidian Revolution"?', 'id': '56de3cd0cffd8e1900b4b6bf', 'answers': [{'text': 'Norman', 'answer_start': 336}, {'text': 'Norman', 'answer_start': 336}, {'text': 'Norman', 'answer_start': 336}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Sybilla of Normandy', 'answer_start': 271}], 'question': 'Who did King David I of Scotland Marry?', 'id': '5ad3f8d2604f3c001a3ffa8d', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Normans and Norman culture', 'answer_start': 324}], 'question': 'What did Sybilla of Normandy introduce to Scotland?', 'id': '5ad3f8d2604f3c001a3ffa8e', 'answers': [], 'is_impossible': True}], 'context': 'Normans came into Scotland, building castles and founding noble families who would provide some future kings, such as Robert the Bruce, as well as founding a considerable number of the Scottish clans. King David I of Scotland, whose elder brother Alexander I had married Sybilla of Normandy, was instrumental in introducing Normans and Norman culture to Scotland, part of the process some scholars call the "Davidian Revolution". Having spent time at the court of Henry I of England (married to David\'s sister Maud of Scotland), and needing them to wrestle the kingdom from his half-brother Máel Coluim mac Alaxandair, David had to reward many with lands. The process was continued under David\'s successors, most intensely of all under William the Lion. The Norman-derived feudal system was applied in varying degrees to most of Scotland. Scottish families of the names Bruce, Gray, Ramsay, Fraser, Ogilvie, Montgomery, Sinclair, Pollock, Burnard, Douglas and Gordon to name but a few, and including the later royal House of Stewart, can all be traced back to Norman ancestry.'}, {'qas': [{'question': 'Where was Ralph earl of?', 'id': '56de3d594396321400ee26ca', 'answers': [{'text': 'Hereford', 'answer_start': 158}, {'text': 'Hereford', 'answer_start': 158}, {'text': 'Hereford', 'answer_start': 158}], 'is_impossible': False}, {'question': 'Who was Ralph in charge of being at war with?', 'id': '56de3d594396321400ee26cb', 'answers': [{'text': 'the Welsh', 'answer_start': 227}, {'text': 'the Welsh', 'answer_start': 227}, {'text': 'the Welsh', 'answer_start': 227}], 'is_impossible': False}, {'question': 'Who made Ralph earl?', 'id': '56de3d594396321400ee26cc', 'answers': [{'text': 'Edward the Confessor', 'answer_start': 90}, {'text': 'Edward the Confessor', 'answer_start': 90}, {'text': 'Edward the Confessor', 'answer_start': 90}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Normans', 'answer_start': 48}], 'question': 'Who came into contact with Wales after the conquest of England?', 'id': '5ad3fb01604f3c001a3ffb35', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Ralph', 'answer_start': 141}], 'question': 'Who made Edward the Confessor Earl?', 'id': '5ad3fb01604f3c001a3ffb36', 'answers': [], 'is_impossible': True}], 'context': 'Even before the Norman Conquest of England, the Normans had come into contact with Wales. Edward the Confessor had set up the aforementioned Ralph as earl of Hereford and charged him with defending the Marches and warring with the Welsh. In these original ventures, the Normans failed to make any headway into Wales.'}, {'qas': [{'question': 'What country was under the control of Norman barons?', 'id': '56de3dbacffd8e1900b4b6d2', 'answers': [{'text': 'Wales', 'answer_start': 299}, {'text': 'Wales', 'answer_start': 299}, {'text': 'Wales', 'answer_start': 299}], 'is_impossible': False}, {'plausible_answers': [{'text': 'the Marches', 'answer_start': 37}], 'question': 'What came under Williams dominace before the conquest?', 'id': '5ad3fb6e604f3c001a3ffb5f', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Bernard de Neufmarché, Roger of Montgomery in Shropshire and Hugh Lupus in Cheshire', 'answer_start': 136}], 'question': 'What Welsh lords did William conquer?', 'id': '5ad3fb6e604f3c001a3ffb60', 'answers': [], 'is_impossible': True}], 'context': "Subsequent to the Conquest, however, the Marches came completely under the dominance of William's most trusted Norman barons, including Bernard de Neufmarché, Roger of Montgomery in Shropshire and Hugh Lupus in Cheshire. These Normans began a long period of slow conquest during which almost all of Wales was at some point subject to Norman interference. Norman words, such as baron (barwn), first entered Welsh at that time."}, {'qas': [{'question': 'What year did Roger de Tosny fail to accomplish what he set out to do?', 'id': '56de3e414396321400ee26d8', 'answers': [{'text': '1018', 'answer_start': 221}, {'text': '1064', 'answer_start': 345}, {'text': '1018', 'answer_start': 221}], 'is_impossible': False}, {'question': 'Who was in charge of the papal army in the War of Barbastro?', 'id': '56de3e414396321400ee26d9', 'answers': [{'text': 'William of Montreuil', 'answer_start': 380}, {'text': 'William of Montreuil', 'answer_start': 380}, {'text': 'William of Montreuil', 'answer_start': 380}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Antioch', 'answer_start': 142}], 'question': 'Where did the Normans carve out a principality before the First Crusade?', 'id': '5ad3fc41604f3c001a3ffb8f', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Reconquista in Iberia', 'answer_start': 195}], 'question': 'What did the Normans take part in in the 10th century?', 'id': '5ad3fc41604f3c001a3ffb90', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Roger de Tosny', 'answer_start': 227}], 'question': 'Who carved out a state for himself from Moorish lands?', 'id': '5ad3fc41604f3c001a3ffb91', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'the War of Barbastro', 'answer_start': 358}], 'question': 'What war occured in the 1oth century?', 'id': '5ad3fc41604f3c001a3ffb92', 'answers': [], 'is_impossible': True}], 'context': 'The legendary religious zeal of the Normans was exercised in religious wars long before the First Crusade carved out a Norman principality in Antioch. They were major foreign participants in the Reconquista in Iberia. In 1018, Roger de Tosny travelled to the Iberian Peninsula to carve out a state for himself from Moorish lands, but failed. In 1064, during the War of Barbastro, William of Montreuil led the papal army and took a huge booty.'}, {'qas': [{'question': 'When did the Siege of Antioch take place?', 'id': '56de3ebc4396321400ee26e6', 'answers': [{'text': '1097', 'answer_start': 267}, {'text': '1097', 'answer_start': 267}, {'text': '1097', 'answer_start': 267}], 'is_impossible': False}, {'question': "What was the name of Bohemond's nephew?", 'id': '56de3ebc4396321400ee26e7', 'answers': [{'text': 'Tancred', 'answer_start': 100}, {'text': 'Tancred', 'answer_start': 100}, {'text': 'Tancred', 'answer_start': 100}], 'is_impossible': False}, {'question': 'What major conquest did Tancred play a roll in?', 'id': '56de3ebc4396321400ee26e8', 'answers': [{'text': 'Jerusalem', 'answer_start': 390}, {'text': 'Jerusalem', 'answer_start': 390}, {'text': 'Jerusalem', 'answer_start': 390}], 'is_impossible': False}, {'plausible_answers': [{'text': '1097', 'answer_start': 267}], 'question': 'When did Tancred lay siege to Antioch?', 'id': '5ad4017a604f3c001a3ffd1f', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Bohemond', 'answer_start': 273}], 'question': "What was the name of Tancred's nephew?", 'id': '5ad4017a604f3c001a3ffd20', 'answers': [], 'is_impossible': True}], 'context': 'In 1096, Crusaders passing by the siege of Amalfi were joined by Bohemond of Taranto and his nephew Tancred with an army of Italo-Normans. Bohemond was the de facto leader of the Crusade during its passage through Asia Minor. After the successful Siege of Antioch in 1097, Bohemond began carving out an independent principality around that city. Tancred was instrumental in the conquest of Jerusalem and he worked for the expansion of the Crusader kingdom in Transjordan and the region of Galilee.[citation needed]'}, {'qas': [{'question': 'How long did Western Europe control Cyprus?', 'id': '56de3efccffd8e1900b4b6fe', 'answers': [{'text': '380 years', 'answer_start': 189}, {'text': '380 years', 'answer_start': 189}, {'text': '380 years', 'answer_start': 189}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Cyprus', 'answer_start': 16}], 'question': 'Who defeated Anglo-Norman forces during the third Crusade?', 'id': '5ad401f2604f3c001a3ffd41', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Cyprus', 'answer_start': 16}], 'question': 'Who dominated Western Europe for 380 years?', 'id': '5ad401f2604f3c001a3ffd42', 'answers': [], 'is_impossible': True}], 'context': 'The conquest of Cyprus by the Anglo-Norman forces of the Third Crusade opened a new chapter in the history of the island, which would be under Western European domination for the following 380 years. Although not part of a planned operation, the conquest had much more permanent results than initially expected.'}, {'qas': [{'question': "What ruined Richard's plans to reach Acre?", 'id': '56de3f784396321400ee26fa', 'answers': [{'text': 'a storm', 'answer_start': 99}, {'text': 'a storm', 'answer_start': 99}, {'text': 'a storm', 'answer_start': 99}], 'is_impossible': False}, {'question': "Who was Richard's fiancee?", 'id': '56de3f784396321400ee26fb', 'answers': [{'text': 'Berengaria', 'answer_start': 218}, {'text': 'Berengaria', 'answer_start': 218}, {'text': 'Berengaria', 'answer_start': 218}], 'is_impossible': False}, {'question': "What year did the storm hit Richard's fleet?", 'id': '56de3f784396321400ee26fc', 'answers': [{'text': '1191', 'answer_start': 9}, {'text': '1191', 'answer_start': 9}, {'text': '1191', 'answer_start': 9}], 'is_impossible': False}, {'question': 'Who ruled Cyprus in 1191?', 'id': '56de3f784396321400ee26fd', 'answers': [{'text': 'Isaac Komnenos', 'answer_start': 421}, {'text': 'Isaac', 'answer_start': 522}, {'text': 'Isaac Komnenos', 'answer_start': 421}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Richard the Lion-hearted', 'answer_start': 14}], 'question': 'Who left Messina in the 11th century?', 'id': '5ad40280604f3c001a3ffd57', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': '1191', 'answer_start': 446}], 'question': 'What year did Richards fleet avoid a storm?', 'id': '5ad40280604f3c001a3ffd58', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Isaac Komnenos', 'answer_start': 421}], 'question': 'Who ruled Cyprus in the 11th century?', 'id': '5ad40280604f3c001a3ffd59', 'answers': [], 'is_impossible': True}], 'context': "In April 1191 Richard the Lion-hearted left Messina with a large fleet in order to reach Acre. But a storm dispersed the fleet. After some searching, it was discovered that the boat carrying his sister and his fiancée Berengaria was anchored on the south coast of Cyprus, together with the wrecks of several other ships, including the treasure ship. Survivors of the wrecks had been taken prisoner by the island's despot Isaac Komnenos. On 1 May 1191, Richard's fleet arrived in the port of Limassol on Cyprus. He ordered Isaac to release the prisoners and the treasure. Isaac refused, so Richard landed his troops and took Limassol."}, {'qas': [{'question': "Who was Guy's Rival?", 'id': '56de40da4396321400ee2708', 'answers': [{'text': 'Conrad of Montferrat', 'answer_start': 188}, {'text': 'Conrad of Montferrat', 'answer_start': 188}, {'text': 'Conrad of Montferrat', 'answer_start': 188}], 'is_impossible': False}, {'question': "What were Isaac's chains made out of?", 'id': '56de40da4396321400ee2709', 'answers': [{'text': 'silver', 'answer_start': 565}, {'text': 'silver', 'answer_start': 565}, {'text': 'silver', 'answer_start': 565}], 'is_impossible': False}, {'question': "Who led Richard's troops when Cyprus was conquered?", 'id': '56de40da4396321400ee270a', 'answers': [{'text': 'Guy de Lusignan', 'answer_start': 85}, {'text': 'Guy de Lusignan', 'answer_start': 508}, {'text': 'Guy de Lusignan', 'answer_start': 508}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Isaac', 'answer_start': 525}], 'question': "Who's chains were made out of copper?", 'id': '5ad404a6604f3c001a3ffde1', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Richard', 'answer_start': 588}], 'question': 'Who led Issacs troops to Cyprus?', 'id': '5ad404a6604f3c001a3ffde2', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Richard', 'answer_start': 658}], 'question': 'Who offered Issac his daughter?', 'id': '5ad404a6604f3c001a3ffde3', 'answers': [], 'is_impossible': True}], 'context': 'Various princes of the Holy Land arrived in Limassol at the same time, in particular Guy de Lusignan. All declared their support for Richard provided that he support Guy against his rival Conrad of Montferrat. The local barons abandoned Isaac, who considered making peace with Richard, joining him on the crusade, and offering his daughter in marriage to the person named by Richard. But Isaac changed his mind and tried to escape. Richard then proceeded to conquer the whole island, his troops being led by Guy de Lusignan. Isaac surrendered and was confined with silver chains, because Richard had promised that he would not place him in irons. By 1 June, Richard had conquered the whole island. His exploit was well publicized and contributed to his reputation; he also derived significant financial gains from the conquest of the island. Richard left for Acre on 5 June, with his allies. Before his departure, he named two of his Norman generals, Richard de Camville and Robert de Thornham, as governors of Cyprus.'}, {'qas': [{'question': 'What continent are the Canarian Islands off the coast of?', 'id': '56de49564396321400ee277a', 'answers': [{'text': 'Africa', 'answer_start': 219}, {'text': 'Africa', 'answer_start': 219}, {'text': 'Africa', 'answer_start': 219}], 'is_impossible': False}, {'plausible_answers': [{'text': 'Jean de Bethencourt and the Poitevine Gadifer de la Salle', 'answer_start': 62}], 'question': 'Who conquered the Canary Island in the 14th century?', 'id': '5ad40419604f3c001a3ffdb7', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Canarian islands', 'answer_start': 134}], 'question': 'What Islands are of the coast of Asia?', 'id': '5ad40419604f3c001a3ffdb8', 'answers': [], 'is_impossible': True}], 'context': 'Between 1402 and 1405, the expedition led by the Norman noble Jean de Bethencourt and the Poitevine Gadifer de la Salle conquered the Canarian islands of Lanzarote, Fuerteventura and El Hierro off the Atlantic coast of Africa. Their troops were gathered in Normandy, Gascony and were later reinforced by Castilian colonists.'}, {'qas': [{'question': 'Who became the King of the Canary Islands?', 'id': '56de49a8cffd8e1900b4b7a7', 'answers': [{'text': 'Bethencourt', 'answer_start': 0}, {'text': 'Bethencourt', 'answer_start': 0}, {'text': 'Bethencourt', 'answer_start': 0}], 'is_impossible': False}, {'question': 'Who bought the rights?', 'id': '56de49a8cffd8e1900b4b7a8', 'answers': [{'text': 'Enrique Pérez de Guzmán', 'answer_start': 172}, {'text': 'Enrique Pérez de Guzmán', 'answer_start': 172}, {'text': 'Enrique Pérez de Guzmán', 'answer_start': 172}], 'is_impossible': False}, {'question': 'Who sold the rights?', 'id': '56de49a8cffd8e1900b4b7a9', 'answers': [{'text': 'Maciot de Bethencourt', 'answer_start': 116}, {'text': 'Maciot de Bethencourt', 'answer_start': 116}, {'text': 'Maciot de Bethencourt', 'answer_start': 116}], 'is_impossible': False}, {'plausible_answers': [{'text': 'King of the Canary Islands', 'answer_start': 30}], 'question': 'What title did Henry II take in the Canary Island?', 'id': '5ad403c1604f3c001a3ffd97', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Maciot de Bethencourt', 'answer_start': 116}], 'question': 'Who sold the rights to the island in the 14th century?', 'id': '5ad403c1604f3c001a3ffd98', 'answers': [], 'is_impossible': True}], 'context': "Bethencourt took the title of King of the Canary Islands, as vassal to Henry III of Castile. In 1418, Jean's nephew Maciot de Bethencourt sold the rights to the islands to Enrique Pérez de Guzmán, 2nd Count de Niebla."}, {'qas': [{'question': 'Where are Jersey and Guernsey', 'id': '56de4a474396321400ee2786', 'answers': [{'text': 'Channel Islands', 'answer_start': 155}, {'text': 'the Channel Islands', 'answer_start': 151}, {'text': 'the Channel Islands', 'answer_start': 151}], 'is_impossible': False}, {'question': 'How many customaries does Norman customary law have?', 'id': '56de4a474396321400ee2787', 'answers': [{'text': 'two', 'answer_start': 212}, {'text': 'two', 'answer_start': 212}, {'text': 'two', 'answer_start': 212}], 'is_impossible': False}, {'plausible_answers': [{'text': 'The customary law of Normandy', 'answer_start': 0}], 'question': 'What Norman law wasdeveloped between 1000 and 1300?', 'id': '5ad40358604f3c001a3ffd7d', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Norman customary law', 'answer_start': 172}], 'question': 'What law has 3 customeries?', 'id': '5ad40358604f3c001a3ffd7e', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Summa de legibus Normanniae in curia laïcali)', 'answer_start': 461}], 'question': 'What was authored in the 12th century?', 'id': '5ad40358604f3c001a3ffd7f', 'answers': [], 'is_impossible': True}], 'context': 'The customary law of Normandy was developed between the 10th and 13th centuries and survives today through the legal systems of Jersey and Guernsey in the Channel Islands. Norman customary law was transcribed in two customaries in Latin by two judges for use by them and their colleagues: These are the Très ancien coutumier (Very ancient customary), authored between 1200 and 1245; and the Grand coutumier de Normandie (Great customary of Normandy, originally Summa de legibus Normanniae in curia laïcali), authored between 1235 and 1245.'}, {'qas': [{'question': 'What is the Norman architecture idiom?', 'id': '56de4a89cffd8e1900b4b7bd', 'answers': [{'text': 'Romanesque', 'answer_start': 135}, {'text': 'Romanesque', 'answer_start': 135}, {'text': 'Romanesque', 'answer_start': 135}], 'is_impossible': False}, {'question': 'What kind of arches does Norman architecture have?', 'id': '56de4a89cffd8e1900b4b7be', 'answers': [{'text': 'rounded', 'answer_start': 332}, {'text': 'rounded', 'answer_start': 332}, {'text': 'rounded', 'answer_start': 332}], 'is_impossible': False}, {'plausible_answers': [{'text': 'rounded arches', 'answer_start': 332}], 'question': 'What type of arch did the Normans invent?', 'id': '5ad402ce604f3c001a3ffd67', 'answers': [], 'is_impossible': True}], 'context': 'Norman architecture typically stands out as a new stage in the architectural history of the regions they subdued. They spread a unique Romanesque idiom to England and Italy, and the encastellation of these regions with keeps in their north French style fundamentally altered the military landscape. Their style was characterised by rounded arches, particularly over windows and doorways, and massive proportions.'}, {'qas': [{'question': 'What architecture type came after Norman in England?', 'id': '56de4b074396321400ee2793', 'answers': [{'text': 'Early Gothic', 'answer_start': 108}, {'text': 'Early Gothic', 'answer_start': 108}, {'text': 'Early Gothic', 'answer_start': 108}], 'is_impossible': False}, {'question': 'What architecture type came before Norman in England?', 'id': '56de4b074396321400ee2794', 'answers': [{'text': 'Anglo-Saxon', 'answer_start': 79}, {'text': 'Anglo-Saxon', 'answer_start': 79}, {'text': 'Anglo-Saxon', 'answer_start': 79}], 'is_impossible': False}, {'question': 'What place had the Norman Arab architectural style?', 'id': '56de4b074396321400ee2795', 'answers': [{'text': 'Sicily', 'answer_start': 328}, {'text': 'Sicily', 'answer_start': 328}, {'text': 'Kingdom of Sicily', 'answer_start': 317}], 'is_impossible': False}, {'plausible_answers': [{'text': 'the period of Norman architecture', 'answer_start': 12}], 'question': 'What precedes the period of Anglo-Saxon architecture?', 'id': '5ad400b0604f3c001a3ffcdf', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Anglo-Saxon', 'answer_start': 79}], 'question': 'What architecture type came after Early Gothic?', 'id': '5ad400b0604f3c001a3ffce0', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Normans', 'answer_start': 145}], 'question': 'Who incorperated Islamic, LOmbard, and Byzantine building techniques in England?', 'id': '5ad400b0604f3c001a3ffce1', 'answers': [], 'is_impossible': True}], 'context': 'In England, the period of Norman architecture immediately succeeds that of the Anglo-Saxon and precedes the Early Gothic. In southern Italy, the Normans incorporated elements of Islamic, Lombard, and Byzantine building techniques into their own, initiating a unique style known as Norman-Arab architecture within the Kingdom of Sicily.'}, {'qas': [{'question': 'When did the church reform begin?', 'id': '56de4b5c4396321400ee2799', 'answers': [{'text': 'early 11th century', 'answer_start': 129}, {'text': '11th century', 'answer_start': 135}, {'text': 'in the early 11th century', 'answer_start': 122}], 'is_impossible': False}, {'question': 'Who used the church to unify themselves?', 'id': '56de4b5c4396321400ee279a', 'answers': [{'text': 'dukes', 'answer_start': 152}, {'text': 'the dukes', 'answer_start': 422}, {'text': 'dukes', 'answer_start': 426}], 'is_impossible': False}, {'plausible_answers': [{'text': 'visual arts', 'answer_start': 7}], 'question': 'What kind of art did the Normans have a rich tradition of?', 'id': '5ad3ffd7604f3c001a3ffca7', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'the dukes', 'answer_start': 148}], 'question': 'Who began a program of church reform in the 1100s', 'id': '5ad3ffd7604f3c001a3ffca8', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'the dukes', 'answer_start': 148}], 'question': 'Who was divided by the church?', 'id': '5ad3ffd7604f3c001a3ffca9', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Normandy', 'answer_start': 859}], 'question': 'Who experienced aa golden age in the 1100s and 1200s', 'id': '5ad3ffd7604f3c001a3ffcaa', 'answers': [], 'is_impossible': True}], 'context': 'In the visual arts, the Normans did not have the rich and distinctive traditions of the cultures they conquered. However, in the early 11th century the dukes began a programme of church reform, encouraging the Cluniac reform of monasteries and patronising intellectual pursuits, especially the proliferation of scriptoria and the reconstitution of a compilation of lost illuminated manuscripts. The church was utilised by the dukes as a unifying force for their disparate duchy. The chief monasteries taking part in this "renaissance" of Norman art and scholarship were Mont-Saint-Michel, Fécamp, Jumièges, Bec, Saint-Ouen, Saint-Evroul, and Saint-Wandrille. These centres were in contact with the so-called "Winchester school", which channeled a pure Carolingian artistic tradition to Normandy. In the final decade of the 11th and first of the 12th century, Normandy experienced a golden age of illustrated manuscripts, but it was brief and the major scriptoria of Normandy ceased to function after the midpoint of the century.'}, {'qas': [{'question': 'When were the French wars of religion?', 'id': '56de4bb84396321400ee27a2', 'answers': [{'text': '16th century', 'answer_start': 35}, {'text': 'the 16th century', 'answer_start': 31}, {'text': 'in the 16th century', 'answer_start': 28}], 'is_impossible': False}, {'plausible_answers': [{'text': 'The French Wars of Religion', 'answer_start': 0}], 'question': 'What wars did France fight in the 1600s?', 'id': '5ad3ff1b604f3c001a3ffc73', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'French Revolution', 'answer_start': 52}], 'question': "What revolution was fought in the 1899's?", 'id': '5ad3ff1b604f3c001a3ffc74', 'answers': [], 'is_impossible': True}], 'context': 'The French Wars of Religion in the 16th century and French Revolution in the 18th successively destroyed much of what existed in the way of the architectural and artistic remnant of this Norman creativity. The former, with their violence, caused the wanton destruction of many Norman edifices; the latter, with its assault on religion, caused the purposeful destruction of religious objects of any type, and its destabilisation of society resulted in rampant pillaging.'}, {'qas': [{'question': 'What kind of needlework was used in the creation of the Bayeux Tapestry?', 'id': '56de4c324396321400ee27ab', 'answers': [{'text': 'embroidery', 'answer_start': 104}, {'text': 'embroidery', 'answer_start': 104}, {'text': 'embroidery', 'answer_start': 104}], 'is_impossible': False}, {'question': "What is Norman art's most well known piece?", 'id': '56de4c324396321400ee27ac', 'answers': [{'text': 'Bayeux Tapestry', 'answer_start': 49}, {'text': 'the Bayeux Tapestry', 'answer_start': 45}, {'text': 'the Bayeux Tapestry', 'answer_start': 45}], 'is_impossible': False}, {'question': 'Who commissioned the Tapestry?', 'id': '56de4c324396321400ee27ad', 'answers': [{'text': 'Odo', 'answer_start': 139}, {'text': 'Odo', 'answer_start': 139}, {'text': 'Odo', 'answer_start': 139}], 'is_impossible': False}, {'plausible_answers': [{'text': 'the Bayeux Tapestry', 'answer_start': 45}], 'question': 'What is the oldest work of Norman art?', 'id': '5ad3fe91604f3c001a3ffc47', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Odo', 'answer_start': 139}], 'question': 'Who commissioned Danish vikings to create the Bayeux Tapestry?', 'id': '5ad3fe91604f3c001a3ffc48', 'answers': [], 'is_impossible': True}], 'context': 'By far the most famous work of Norman art is the Bayeux Tapestry, which is not a tapestry but a work of embroidery. It was commissioned by Odo, the Bishop of Bayeux and first Earl of Kent, employing natives from Kent who were learned in the Nordic traditions imported in the previous half century by the Danish Vikings.'}, {'qas': [{'question': 'What is the most important type of Norman art preserved in churches?', 'id': '56de51244396321400ee27ef', 'answers': [{'text': 'mosaics', 'answer_start': 466}, {'text': 'mosaics', 'answer_start': 466}, {'text': 'mosaics', 'answer_start': 466}], 'is_impossible': False}, {'plausible_answers': [{'text': 'as stonework or metalwork', 'answer_start': 42}], 'question': 'How has British art survived in Normandy?', 'id': '5ad3fe0d604f3c001a3ffc1b', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'mosaics', 'answer_start': 466}], 'question': 'What is the most common form of Norman art in churches?', 'id': '5ad3fe0d604f3c001a3ffc1c', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'Lombard Salerno', 'answer_start': 549}], 'question': 'What was a centre of ivorywork in the 1100s?', 'id': '5ad3fe0d604f3c001a3ffc1d', 'answers': [], 'is_impossible': True}], 'context': 'In Britain, Norman art primarily survives as stonework or metalwork, such as capitals and baptismal fonts. In southern Italy, however, Norman artwork survives plentifully in forms strongly influenced by its Greek, Lombard, and Arab forebears. Of the royal regalia preserved in Palermo, the crown is Byzantine in style and the coronation cloak is of Arab craftsmanship with Arabic inscriptions. Many churches preserve sculptured fonts, capitals, and more importantly mosaics, which were common in Norman Italy and drew heavily on the Greek heritage. Lombard Salerno was a centre of ivorywork in the 11th century and this continued under Norman domination. Finally should be noted the intercourse between French Crusaders traveling to the Holy Land who brought with them French artefacts with which to gift the churches at which they stopped in southern Italy amongst their Norman cousins. For this reason many south Italian churches preserve works from France alongside their native pieces.'}, {'qas': [{'question': 'In what century did important classical music developments occur in Normandy?', 'id': '56de51c64396321400ee27f7', 'answers': [{'text': '11th', 'answer_start': 97}, {'text': 'the 11th', 'answer_start': 93}, {'text': '11th', 'answer_start': 97}], 'is_impossible': False}, {'question': 'Who were the two abbots at Fécamp Abbey?', 'id': '56de51c64396321400ee27f8', 'answers': [{'text': 'William of Volpiano and John of Ravenna', 'answer_start': 234}, {'text': 'William of Volpiano and John of Ravenna', 'answer_start': 234}, {'text': 'William of Volpiano and John of Ravenna', 'answer_start': 234}], 'is_impossible': False}, {'plausible_answers': [{'text': 'classical music', 'answer_start': 74}], 'question': 'What developed in Normandy during the 1100s?', 'id': '5ad3fd68604f3c001a3ffbe7', 'answers': [], 'is_impossible': True}, {'plausible_answers': [{'text': 'musical composition', 'answer_start': 632}], 'question': 'What was Fecamp Abby the center of?', 'id': '5ad3fd68604f3c001a3ffbe8', 'answers': [], 'is_impossible': True}], 'context': 'Normandy was the site of several important developments in the history of classical music in the 11th century. Fécamp Abbey and Saint-Evroul Abbey were centres of musical production and education. At Fécamp, under two Italian abbots, William of Volpiano and John of Ravenna, the system of denoting notes by letters was developed and taught. It is still the most common form of pitch representation in English- and German-speaking countries today. Also at Fécamp, the staff, around which neumes were oriented, was first developed and taught in the 11th century. Under the German abbot Isembard, La Trinité-du-Mont became a centre of musical composition.'}, {'qas': [{'question': 'Where did the monks flee to?', 'id': '56de52614396321400ee27fb', 'answers': [{'text': 'southern Italy', 'answer_start': 179}, {'text': 'southern Italy', 'answer_start': 179}, {'text': 'southern Italy', 'answer_start': 179}], 'is_impossible': False}, {'question': 'What monastery did the Saint-Evroul monks establish in Italy?', 'id': '56de52614396321400ee27fc', 'answers': [{'text': "Latin monastery at Sant'Eufemia.", 'answer_start': 259}, {'text': "a Latin monastery at Sant'Eufemia", 'answer_start': 257}, {'text': "Sant'Eufemia", 'answer_start': 278}], 'is_impossible': False}, {'question': 'Who patronized the monks in Italy? ', 'id': '56de52614396321400ee27fd', 'answers': [{'text': 'Robert Guiscard', 'answer_start': 225}, {'text': 'Robert Guiscard', 'answer_start': 225}, {'text': 'Robert Guiscard', 'answer_start': 225}], 'is_impossible': False}, {'question': 'What tradition were the Saint-Evroul monks known for?', 'id': '56de52614396321400ee27fe', 'answers': [{'text': 'singing', 'answer_start': 32}, {'text': 'singing', 'answer_start': 32}, {'text': 'singing', 'answer_start': 330}], 'is_impossible': False}, {'plausible_answers': [{'text': 'monks', 'answer_start': 149}], 'question': 'Who fled from southern Italy?', 'id': '5ad3fccf604f3c001a3ffbb5', 'answers': [], 'is_impossible': True}], 'context': "At Saint Evroul, a tradition of singing had developed and the choir achieved fame in Normandy. Under the Norman abbot Robert de Grantmesnil, several monks of Saint-Evroul fled to southern Italy, where they were patronised by Robert Guiscard and established a Latin monastery at Sant'Eufemia. There they continued the tradition of singing."}]}]
#     input_data_main = {"data": [{"title": "Beyonc\u00e9", "paragraphs": [{"qas": [{"question": "When did Beyonce start becoming popular?", "id": "56be85543aeaaa14008c9063", "answers": [{"text": "in the late 1990s", "answer_start": 269}], "is_impossible": False}, {"question": "What areas did Beyonce compete in when she was growing up?", "id": "56be85543aeaaa14008c9065", "answers": [{"text": "singing and dancing", "answer_start": 207}], "is_impossible": False}, {"question": "When did Beyonce leave Destiny's Child and become a solo singer?", "id": "56be85543aeaaa14008c9066", "answers": [{"text": "2003", "answer_start": 526}], "is_impossible": False}, {"question": "In what city and state did Beyonce  grow up? ", "id": "56bf6b0f3aeaaa14008c9601", "answers": [{"text": "Houston, Texas", "answer_start": 166}], "is_impossible": False}, {"question": "In which decade did Beyonce become famous?", "id": "56bf6b0f3aeaaa14008c9602", "answers": [{"text": "late 1990s", "answer_start": 276}], "is_impossible": False}, {"question": "In what R&B group was she the lead singer?", "id": "56bf6b0f3aeaaa14008c9603", "answers": [{"text": "Destiny's Child", "answer_start": 320}], "is_impossible": False}, {"question": "What album made her a worldwide known artist?", "id": "56bf6b0f3aeaaa14008c9604", "answers": [{"text": "Dangerously in Love", "answer_start": 505}], "is_impossible": False}, {"question": "Who managed the Destiny's Child group?", "id": "56bf6b0f3aeaaa14008c9605", "answers": [{"text": "Mathew Knowles", "answer_start": 360}], "is_impossible": False}, {"question": "When did Beyonc\u00e9 rise to fame?", "id": "56d43c5f2ccc5a1400d830a9", "answers": [{"text": "late 1990s", "answer_start": 276}], "is_impossible": False}, {"question": "What role did Beyonc\u00e9 have in Destiny's Child?", "id": "56d43c5f2ccc5a1400d830aa", "answers": [{"text": "lead singer", "answer_start": 290}], "is_impossible": False}, {"question": "What was the first album Beyonc\u00e9 released as a solo artist?", "id": "56d43c5f2ccc5a1400d830ab", "answers": [{"text": "Dangerously in Love", "answer_start": 505}], "is_impossible": False}, {"question": "When did Beyonc\u00e9 release Dangerously in Love?", "id": "56d43c5f2ccc5a1400d830ac", "answers": [{"text": "2003", "answer_start": 526}], "is_impossible": False}, {"question": "How many Grammy awards did Beyonc\u00e9 win for her first solo album?", "id": "56d43c5f2ccc5a1400d830ad", "answers": [{"text": "five", "answer_start": 590}], "is_impossible": False}, {"question": "What was Beyonc\u00e9's role in Destiny's Child?", "id": "56d43ce42ccc5a1400d830b4", "answers": [{"text": "lead singer", "answer_start": 290}], "is_impossible": False}, {"question": "What was the name of Beyonc\u00e9's first solo album?", "id": "56d43ce42ccc5a1400d830b5", "answers": [{"text": "Dangerously in Love", "answer_start": 505}], "is_impossible": False}], "context": "Beyonc\u00e9 Giselle Knowles-Carter (/bi\u02d0\u02c8j\u0252nse\u026a/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyonc\u00e9's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles \"Crazy in Love\" and \"Baby Boy\"."}]}]}
#     input_data = input_data_main['data']
#     def is_whitespace(c):
#         if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
#             return True
#         return False
#
#     examples = []
#     for entry in input_data:
#         for paragraph in entry["paragraphs"]:
#             paragraph_text = paragraph["context"]
#             doc_tokens = []
#             char_to_word_offset = []
#             prev_is_whitespace = True
#             for c in paragraph_text:
#                 if is_whitespace(c):
#                     prev_is_whitespace = True
#                 else:
#                     if prev_is_whitespace:
#                         doc_tokens.append(c)
#                     else:
#                         doc_tokens[-1] += c
#                     prev_is_whitespace = False
#                 char_to_word_offset.append(len(doc_tokens) - 1)
#
#             for qa in paragraph["qas"]:
#                 qas_id = qa["id"]
#                 question_text = qa["question"]
#                 start_position = None
#                 end_position = None
#                 orig_answer_text = None
#                 is_impossible = False
#                 if is_training:
#                     if version_2_with_negative:
#                         is_impossible = qa["is_impossible"]
#                     if (len(qa["answers"]) != 1) and (not is_impossible):
#                         raise ValueError(
#                             "For training, each question should have exactly 1 answer.")
#                     if not is_impossible:
#                         answer = qa["answers"][0]
#                         orig_answer_text = answer["text"]
#                         answer_offset = answer["answer_start"]
#                         answer_length = len(orig_answer_text)
#                         start_position = char_to_word_offset[answer_offset]
#                         end_position = char_to_word_offset[answer_offset + answer_length - 1]
#                         actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
#                         cleaned_answer_text = " ".join(
#                             whitespace_tokenize(orig_answer_text))
#                         if actual_text.find(cleaned_answer_text) == -1:
#                             print("Could not find answer: '%s' vs. '%s'",
#                                            actual_text, cleaned_answer_text)
#                             continue
#                     else:
#                         start_position = -1
#                         end_position = -1
#                         orig_answer_text = ""
#
#                 example = SquadExample(
#                     qas_id=qas_id,
#                     question_text=question_text,
#                     doc_tokens=doc_tokens,
#                     orig_answer_text=orig_answer_text,
#                     start_position=start_position,
#                     end_position=end_position,
#                     is_impossible=is_impossible)
#                 examples.append(example)
#     return examples
def read_squad_examples(input_file, is_training= False, version_2_with_negative=False):
    """Read a SQuAD json file into a list of SquadExample."""
#     with open(input_file, "r", encoding='utf-8') as reader:
#         input_data = json.load(reader)["data"]
    input_data = input_file
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
#                             logger.warning("Could not find answer: '%s' '%s' vs. '%s'",
#                                            qas_id, actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples
def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        _DocSpan = collections.namedtuple(
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0
            # if example_index < 20:
            #     print("*** Example ***")
            #     print("unique_id: %s" % (unique_id))
            #     print("example_index: %s" % (example_index))
            #     print("doc_span_index: %s" % (doc_span_index))
            #     print("tokens: %s" % " ".join(tokens))
            #     print("token_to_orig_map: %s" % " ".join([
            #         "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
            #     print("token_is_max_context: %s" % " ".join([
            #         "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
            #     ]))
            #     print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     print(
            #         "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     print(
            #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #     if is_training and example.is_impossible:
            #         print("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    print("start_position: %d" % (start_position))
                    print("end_position: %d" % (end_position))
                    print(
                        "answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible))
            unique_id += 1

    return features



def read_examples(examples, is_training = True):
    ds = []
    import json
    input_data = examples
    # with open(input_file, "r", encoding='utf-8') as reader:
    #     input_data = json.load(reader)
    print(f'no of examples : {len(input_data)}')


    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    nominalized_emotion = {
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "joy": "joy",
        "sadness": "sadness",
        "surprise": "surprise"
    }
    for example in input_data:
        for i, current_utt in enumerate(example['conversation']):
            current_emotion = current_utt['emotion']
            if current_emotion == 'neutral':
                continue
            current_utterance = current_utt['text']
            current_speaker = current_utt['speaker']
            current_utterance_id = current_utt['utterance_ID']
            previous_conversations = example['conversation'][:i] if i > 0 else []

            conversation_until_now_arr = []
            if previous_conversations:

                for pc in previous_conversations:
                    speaker = pc['speaker']
                    txt = pc['text']
                    conversation_until_now_arr.append(f'{speaker}: {txt}')
                # assumming full-stops exist after each utterance
            conversation_until_now_arr.append(f'{current_speaker}: {current_utterance}')
            conversation_until_now = " ".join(conversation_until_now_arr)
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in conversation_until_now:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        if doc_tokens:
                            doc_tokens[-1] += c
                        else:
                            doc_tokens.append( c)
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            # sample = f'[CLS]{conversation_until_now}[CURRENT_UTT]{current_utterance}[SEP]What caused the emotion of {current_emotion} in the current utterance?'
            question_prompt = f"The current utterance is - {current_utterance}. What caused the {nominalized_emotion[current_emotion]} in the current utterance?"
            #sample2 = f'{conversation_until_now}[SEP]{question_prompt}'
            # print(self.tokenizer(sample2))
            #print(sample2)
            if is_training:
                labels_arr = []
                labels_arr = _get_emotion_cause_labels(example = example , current_emotion= current_emotion, current_conv_id= current_utterance_id, conversation = conversation_until_now)
                #start_position, end_position = _get_emotion_cause_labels(example, current_emotion ,current_utterance, i)
                for labels in labels_arr:
                    start_offset = labels[0]
                    end_offset = labels[1]

                    start_position = char_to_word_offset[start_offset]
                    end_position = char_to_word_offset[end_offset-1]
                    actual_text = " ".join(doc_tokens[start_position:(end_position+1)])
                    orig_answer_text = labels[2]
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        print("Could not find answer: '%s' vs. '%s'",
                                       actual_text, cleaned_answer_text)
                        raise Exception(f"Error reading example {current_utterance_id}")

                    qexample = SquadExample(
                        qas_id=f"{example['conversation_ID']}_{i}",
                        question_text=question_prompt,
                        doc_tokens=doc_tokens,
                        orig_answer_text=conversation_until_now,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=False)
                    ds.append(qexample)
            else:
                qexample = SquadExample(
                    qas_id=f"{example['conversation_ID']}_{i}",
                    question_text=question_prompt,
                    doc_tokens=doc_tokens,
                    orig_answer_text=conversation_until_now,
                    start_position=-1,
                    end_position=-1,
                    is_impossible=False)
                ds.append(qexample)

    return ds

def _get_emotion_cause_labels(example , current_emotion, current_conv_id, conversation):
    # if conversation =="" or conversation == None:
    #     print(f"empty conv {example}")
    labels_in_conversation = example['emotion-cause_pairs']
    selected_labels = []
    for i in labels_in_conversation:
        if i[0] == f'{current_conv_id}_{current_emotion}':
            selected_labels.append(i[1])
    possible_cause_labels = []
    for j in selected_labels:
        causelabel = j.split("_")
        substring = causelabel[1]
        larger_string = conversation
        start = larger_string.find(substring)
        if start != -1:
            end = start + len(substring)
            possible_cause_labels.append( [start, end, causelabel[1]])
        else:
            print(f'conv {example["conversation_ID"]}, label {j} has a case of anticipation')
            #print(f'start {start} ,ls: {larger_string} ,ss: {substring} ,conversation: {conversation}')
            #raise Exception(f"some issue in finding label {substring} in {example['conversation_ID']}")

    return possible_cause_labels
    #data_point looks like
    # {
    #     "conversation_ID": 1,
    #     "conversation": [
    #         {
    #             "utterance_ID": 1,
    #             "text": "Alright , so I am back in high school , I am standing in the middle of the cafeteria , and I realize I am totally naked .",
    #             "speaker": "Chandler",
    #             "emotion": "neutral"
    #         },
    #         {
    #             "utterance_ID": 2,
    #             "text": "Oh , yeah . Had that dream .",
    #             "speaker": "All",
    #             "emotion": "neutral"
    #         },
    #         {
    #             "utterance_ID": 3,
    #             "text": "Then I look down , and I realize there is a phone ... there .",
    #             "speaker": "Chandler",
    #             "emotion": "surprise"
    #         },
    #         {
    #             "utterance_ID": 4,
    #             "text": "Instead of ... ?",
    #             "speaker": "Joey",
    #             "emotion": "surprise"
    #         },
    #         {
    #             "utterance_ID": 5,
    #             "text": "That is right .",
    #             "speaker": "Chandler",
    #             "emotion": "anger"
    #         },
    #         {
    #             "utterance_ID": 6,
    #             "text": "Never had that dream .",
    #             "speaker": "Joey",
    #             "emotion": "neutral"
    #         },
    #         {
    #             "utterance_ID": 7,
    #             "text": "No .",
    #             "speaker": "Phoebe",
    #             "emotion": "neutral"
    #         },
    #         {
    #             "utterance_ID": 8,
    #             "text": "All of a sudden , the phone starts to ring .",
    #             "speaker": "Chandler",
    #             "emotion": "neutral"
    #         }
    #     ],
    #     "emotion-cause_pairs": [
    #         [
    #             "3_surprise",
    #             "1_I realize I am totally naked ."
    #         ],
    #         [
    #             "3_surprise",
    #             "3_Then I look down , and I realize there is a phone ... there ."
    #         ],
    #         [
    #             "4_surprise",
    #             "1_I realize I am totally naked ."
    #         ],
    #         [
    #             "4_surprise",
    #             "3_Then I look down , and I realize there is a phone ... there ."
    #         ],
    #         [
    #             "4_surprise",
    #             "4_Instead of ..."
    #         ],
    #         [
    #             "5_anger",
    #             "1_I realize I am totally naked ."
    #         ],
    #         [
    #             "5_anger",
    #             "3_Then I look down , and I realize there is a phone ... there ."
    #         ],
    #         [
    #             "5_anger",
    #             "4_Instead of ..."
    #         ]
    #     ]
    # }


#converts our dataset to SQuAD style inputs
def convert_to_squad(input_data):

    nominalized_emotion = {
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "joy": "joy",
        "sadness": "sadness",
        "surprise": "surprise"
    }

    data = []
    data_obj = {}
    data_obj['title'] = "SemEval-3"
    data_obj['paragraphs'] = []
    for example in input_data:
        idx_prefix = int(example['conversation_ID']) * 10000
        for i, current_utt in enumerate(example['conversation']):
            current_emotion = current_utt['emotion']
            if current_emotion == 'neutral':
                continue
            current_utterance = current_utt['text']
            current_speaker = current_utt['speaker']
            current_utterance_id = current_utt['utterance_ID']
            previous_conversations = example['conversation'][:i] if i > 0 else []
            question_prompt = f"The current utterance is - {current_utterance}. What caused the {nominalized_emotion[current_emotion]} in the current utterance?"
            conversation_until_now_arr = []
            if previous_conversations:
                for pc in previous_conversations:
                    speaker = pc['speaker']
                    txt = pc['text']
                    conversation_until_now_arr.append(f'{speaker}: {txt}')
                # assumming full-stops exist after each utterance
            conversation_until_now_arr.append(f'{current_speaker}: {current_utterance}')
            conversation_until_now = " ".join(conversation_until_now_arr)
            labels_arr = _get_emotion_cause_labels(example=example, current_emotion=current_emotion,
                                                   current_conv_id=current_utterance_id,
                                                   conversation=conversation_until_now)
            for label_entry in labels_arr:
                obj = {}
                obj['context'] = conversation_until_now
                qas = {}
                qas['question'] = question_prompt
                qas['id'] = idx_prefix + current_utterance_id
                qas['is_impossibe'] = False
                qas['answers'] = [{"text": label_entry[2], "answer_start": label_entry[0]}]
                obj['qas'] = [qas]
                data_obj['paragraphs'].append(obj)
    data.append(data_obj)

    return {"data": data}




if __name__ == '__main__':
    pass
    # train_examples = read_examples("../data/raw/SemEval-2024_Task3/text/Subtask_1_2_train.json", False)
    # #train_examples = read_example("raw/SemEval-2024_Task3/text/random_trial.json", True)
    # print(f'no of PROCESSED examples : {len(train_examples)}')
    # print(train_examples[0])
    # max_seq_length = 512
    # doc_stride = 128
    # max_query_length = 64
    # train_batch_size = 32
    # model_name = 'SpanBERT/spanbert-base-cased'
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # train_features = convert_examples_to_features(
    #     examples=train_examples,
    #     tokenizer=tokenizer,
    #     max_seq_length=max_seq_length,
    #     doc_stride=doc_stride,
    #     max_query_length=max_query_length,
    #     is_training=False)
    # print(len(train_features))