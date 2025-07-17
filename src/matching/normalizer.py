import re
import pandas as pd


class AddressNormalizer:
    def __init__(self):
        self.aliases = {
            "Խ. ՀԱՅՐԻԿ": "ԽՐԻՄՅԱՆ ՀԱՅՐԻԿԻ",
            "ԽՐԻՄՅԱՆ ՀԱՅՐԻԿ": "ԽՐԻՄՅԱՆ ՀԱՅՐԻԿԻ",
        }

        self.armenian_suffixes = [
            r'\bԽՃՂ\.?', r'\bՃՂ\.?', r'\bՓ\.?', r'\bՊՈՂ\.?', r'\bԱՎ\.?', r'\bՃԱՄԲ\.?', r'\bԹԵԼԱ\.?'
        ]

        # Normalize old and new keys in this mapping
        self.old_to_new_map = {
            self._norm("Ֆրունզեի"): self._norm("Լ. Մադոյան"),
            self._norm("Լենինգրադյան"): self._norm("Վ. Սարգսյան"),
            self._norm("Կիրովականյան"): self._norm("Վանաձորի"),
            self._norm("Կալինինի"): self._norm("Գ. Նժդեհի"),
            self._norm("Կինգիսեպի"): self._norm("Վ. Չերազի"),
            self._norm("Պլեխանովի"): self._norm("Սահմանապահների"),
            self._norm("Շինարարների"): self._norm("Մ. Թետչերի"),
            self._norm("Կիրովի"): self._norm("Ն. Ռիժկովի"),
            self._norm("Լենինի"): self._norm("Տիգրան Մեծի"),
            self._norm("Խ. Հայրիկ"): self._norm("Խրիմյան Հայրիկի"),
            self._norm("Անի թաղամաս Մ. Ավետիսյան"): self._norm("Մ. Ավետիսյան"),
            self._norm("Մարքսի"): self._norm("Պ. Ջափարիձեի"),
            self._norm("Անի թաղամաս Ա. Շահինյան"): self._norm("Ա. Շահինյան"),
            self._norm("Օղակային"): self._norm("Արևելյան շրջանցող"),
            self._norm("Ռեպինի"): self._norm("Բ. Շչերբինայի"),
            self._norm("Հեղափոխության"): self._norm("Գ. Նժդեհի"),
            self._norm("Անի թաղամաս Ե. Չարենցի"): self._norm("Ե. Չարենցի"),
            self._norm("Ղուկասյան փողոց 10-րդ"): self._norm("Յ. Վարդանյան"),
            self._norm("Ղուկասյան փողոց 15-րդ"): self._norm("Յ. Վարդանյան"),
            self._norm("Ղուկասյան փողոց 11-րդ"): self._norm("Յ. Վարդանյան"),
            self._norm("Ղուկասյան փողոց 12-րդ"): self._norm("Յ. Վարդանյան"),
            self._norm("Ղուկասյան փողոց 13-րդ"): self._norm("Յ. Վարդանյան"),
            self._norm("Ղուկասյան փողոց 14-րդ"): self._norm("Յ. Վարդանյան"),
            self._norm("Սևյան"): self._norm("Հ. Ղանդիլյան"),
            self._norm("Մուշ-2  թաղամասի փողոցներից մեկը"): self._norm("Կ. Հալաբյան"),
            self._norm("Ղուկասյան"): self._norm("Յ. Վարդանյան"),
            self._norm("Խաղաղության"): self._norm("Բագրատունյաց"),
            self._norm("Մարքսի"): self._norm("Ջիվանու"),
            self._norm("Ազիզբեկովի"): self._norm("Ն. Շնորհալու"),
            self._norm("Էլեկտրո պրիբորնի 6-րդ շարք"): self._norm("Ա. Արմենյան փողոց"),
            self._norm("Էլեկտրո պրիբորնի 10-րդ շարք"): self._norm("Ա. Գևորգյան փողոց"),
            self._norm("Կիրովաբադյան փողոց"): self._norm("Ա. Թամանյան փողոց"),
            self._norm("50 ամյակի անվան փողոց"): self._norm("Ա. Մանուկյան փողոց"),
            self._norm("<<Անի>> թաղամաս 3-րդ փողոց"): self._norm("Ա. Շահինյան փողոց"),
            self._norm("Հնոցավան 2-րդ շարք"): self._norm("Ա. Պետրոսյան փողոց"),
            self._norm("Կոմսոմոլի փողոց"): self._norm("Ա. Վասիլյան փողոց"),
            self._norm("Կեցխովելի փողոց"): self._norm("Արտակ եպիսկոպոս Սմբատյան փողոց"),
            self._norm("Արվելաձե փողոց"): self._norm("Գարեգին Ա-ի փողոց"),
            self._norm("Էլեկտրո պրիբորնի 8-րդ շարք"): self._norm("Թ. Մանդալյան փողոց"),
            self._norm("Պողպատավան 3-րդ շարք"): self._norm("Ժ. Բ. Բարոնյան փողոց"),
            self._norm("Կրուպսկայա փողոց"): self._norm("Խ. Դաշտենցի փողոց"),
            self._norm("Քութաիսյան փողոց"): self._norm("Կ. Դեմիրճյան փողոց"),
            self._norm("Պողպատավան 2-րդ շարք"): self._norm("Կ. Խաչատրյան փողոց"),
            self._norm("Կույբիշևի փողոց"): self._norm("Հ. Մազմանյան փողոց"),
            self._norm("Պիոներական փողոց"): self._norm("Հ. Մելքոնյան փողոց"),
            self._norm("Պողպատավան 1-ին շարք"): self._norm("Հ. Պողոսյան փողոց"),
            self._norm("Պողպատավան 4-րդ շարք"): self._norm("Հ. Ռասկատլյան փողոց"),
            self._norm("Կատելնայա"): self._norm("Հնոցավանի 1-ին շարք"),
            self._norm("Պետ բարակներ"): self._norm("Ղ. Ղուկասյան փողոց"),
            self._norm("Մայիսյան փողոց"): self._norm("Մ. Մկրտչյան փողոց"),
            self._norm("Էլեկտրո պրիբորնի 7-րդ շարք"): self._norm("Մ. Սարգսյան փողոց"),
            self._norm("Սվերդլովի փողոց"): self._norm("Ն. Ղորղանյան փողոց"),
            self._norm("Աստղի հրապարակ"): self._norm("Շ. Ազնավուրի հրապարակ"),
            self._norm("Ս. Մուսայելյան փողոց"): self._norm("Շ. Ազնավուրի հրապարակ"),
            self._norm("Էլեկտրո պրիբորնի 11-րդ շարք"): self._norm("Ռ. Դանիելյան փողոց"),
            self._norm("Օրջոնիկիձեի փողոց"): self._norm("Ս. Մատնիշյան փողոց"),
            self._norm("Էնգելսի փողոց"): self._norm("Վ. Աճեմյան փողոց"),
            self._norm("Կենտրոնական հրապարակ"): self._norm("Վարդանանց հրապարակ"),
            self._norm("<<Անի>> թաղամաս 15-րդ փողոց"): self._norm("Ֆորալբերգի փողոց"),
        }

    def _norm(self, text):
        text = str(text).strip().upper()
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text)

    def normalize(self, text):
        if pd.isna(text):
            return ""

        text = str(text).strip().upper()

        # Apply direct alias replacement first
        if text in self.aliases:
            text = self.aliases[text]

        # Remove suffixes
        for suffix in self.armenian_suffixes:
            text = re.sub(suffix, '', text, flags=re.IGNORECASE)

        # Remove special chars, extra spaces
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)

        # Remove trailing "Ի" if it's the last letter of each word
        text = ' '.join([w[:-1] if w.endswith("Ի") else w for w in text.split()])

        # Normalize again and map old names to new ones
        text_norm = self._norm(text)
        return self.old_to_new_map.get(text_norm, text_norm)