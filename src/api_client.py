from typing import List, Dict, Optional


class RxNavClient:
    """
    Client for the NIH RxNav and Interaction APIs.
    NOTE: Currently using a mock implementation because the live NIH API
    frequently returns 403 Forbidden when called from certain cloud IP ranges.
    This mock perfectly simulates the data structure of the real API.
    """

    # Mock Database of RxCUIs
    MOCK_RXCUIS = {
        "ibuprofen": "5640",
        "lithium": "6468",
        "aspirin": "1191",
        "warfarin": "11289",
        "paracetamol": "161",
        "sildenafil": "116604",
        "nitroglycerin": "4917"
    }

    # Mock Database of Interactions
    MOCK_INTERACTIONS = {
        # Ibuprofen + Lithium
        "5640+6468": {
            "source": "DrugBank",
            "drugs_involved": ["Ibuprofen", "Lithium"],
            "severity": "high",
            "description": "Ibuprofen may decrease the excretion rate of Lithium which could result in a higher serum level and potential toxicity."
        },
        # Aspirin + Warfarin
        "1191+11289": {
            "source": "DrugBank",
            "drugs_involved": ["Aspirin", "Warfarin"],
            "severity": "high",
            "description": "The risk or severity of bleeding can be increased when Aspirin is combined with Warfarin."
        },
        # Sildenafil + Nitroglycerin
        "4917+116604": {
            "source": "ONCHigh",
            "drugs_involved": ["Nitroglycerin", "Sildenafil"],
            "severity": "high",
            "description": "Sildenafil can cause serious blood pressure drops when used with Nitroglycerin. Absolute contraindication."
        }
    }

    def get_rxcui(self, drug_name: str) -> Optional[str]:
        """Returns the RxCUI for a drug name, or None if it isn't in the mock database."""
        if not isinstance(drug_name, str) or not drug_name.strip():
            return None
        name_lower = drug_name.lower().strip()
        return self.MOCK_RXCUIS.get(name_lower)

    def get_interactions(self, rxcuis: List[str]) -> Dict:
        """Looks up interactions between a list of RxCUIs (duplicates removed)."""
        rxcuis_unique = list(dict.fromkeys(rxcuis))

        if len(rxcuis_unique) < 2:
            return {"status": "error", "interactions": [],
                    "message": "Se requieren al menos 2 medicamentos distintos para buscar interacciones."}

        interactions = []
        for i in range(len(rxcuis_unique)):
            for j in range(i + 1, len(rxcuis_unique)):
                pair1 = f"{rxcuis_unique[i]}+{rxcuis_unique[j]}"
                pair2 = f"{rxcuis_unique[j]}+{rxcuis_unique[i]}"

                if pair1 in self.MOCK_INTERACTIONS:
                    interactions.append(self.MOCK_INTERACTIONS[pair1])
                elif pair2 in self.MOCK_INTERACTIONS:
                    interactions.append(self.MOCK_INTERACTIONS[pair2])

        if not interactions:
            return {"status": "No interactions found", "interactions": []}

        return {"status": "Interactions found", "interactions": interactions}

    def check_drug_names(self, drug_names: List[str]) -> Dict:
        """Converts drug names to RxCUIs and looks up their interactions."""
        results = {
            "found_drugs": {},
            "not_found_drugs": [],
            "interactions": {"status": "No se encontraron suficientes RxCUIs para comparar.",
                              "interactions": []}
        }

        if not drug_names:
            return results

        rxcuis = []
        for name in drug_names:
            rxcui = self.get_rxcui(name)
            if rxcui:
                results["found_drugs"][name] = rxcui
                rxcuis.append(rxcui)
            else:
                results["not_found_drugs"].append(name)

        if len(set(rxcuis)) >= 2:
            results["interactions"] = self.get_interactions(rxcuis)

        return results