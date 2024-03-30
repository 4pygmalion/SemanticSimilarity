from __future__ import annotations

import re
import random
from typing import Set, Dict, Optional, NewType, List
from dataclasses import dataclass, field
import pronto


class HPOIdFormatError(Exception):
    pass


@dataclass(frozen=True)
class Phenotype:
    """HPO개념에 해당하는 ID.

    Note:
        HPO의 Term identifier 은 HP:7자리 숫자로 이루어짐
        예) HP:0000494
    """

    id: str
    name: Optional[str] = None

    def __eq__(self, other):
        if not isinstance(other, Phenotype):
            return False

        if self.id == other.id:
            return True

        return False

    def __hash__(self):
        return hash(self.id)

    def _check_valid_hpo_pattern(self) -> None:
        """HPO의 ID가 "HP:7자리숫자"로 구성되어있는지 확인하는 메서드

        Raises:
            HPOIdFormatError: "HP:7자리숫자"로 id가 전달되지 않는 경우
        """
        if re.fullmatch(r"HP\:\d{7}", self.id) is None:
            raise HPOIdFormatError(f"Passed ID({self.id}) not valid")

        return

    def __post_init__(self):
        self._check_valid_hpo_pattern()

    def check_id_in_ontology(self, ontology: pronto.Ontology) -> bool:
        """HPO에 저장된 ID가 전체 HPO ontology에 정의되어 있는지 확인하는 메서드
        HPO.obo(all_hpo_id)에

        Args:
            all_hpo_id (set): 전체 HPO_ID의 집합

        Returns:
            bool: ontology내에 ID가 존재여부
        """
        if self.id not in ontology.keys():
            return False

        return True

    def is_member_of(self, another_phenos: Set[str]) -> bool:
        return True if self.id in another_phenos else False


@dataclass
class Disease:
    disease_name: str
    disease_id: str
    pheno: Set[Phenotype]

    def get_n_phenos_randomly(self, max_n: int = 10) -> Set[Phenotype]:
        n = random.randint(1, min(len(self.pheno), max_n))
        return set(random.sample(self.pheno, n))

    def to_json(self) -> Dict[str, Disease]:
        dict_obj = self.__dict__.copy()
        dict_obj["pheno"] = [pheno.__dict__ for pheno in dict_obj["pheno"]]
        return dict_obj


@dataclass
class Gene:
    """Gene(유전자)에 해당하는 데이터클레스

    Gene 데이터 클레스 내, Gene에 연관된 질환들의 집합, Gene에 관련된 표현형의 집합을
    속성으로 가지고 있음

    """

    symbol: str  # entrez_gene_symbol
    hpos: Set[Phenotype] = field(default_factory=set)
    disease_ids: Set[str] = field(default_factory=set)

    def __eq__(self, other) -> bool:
        if not (self.symbol == other.symbol and self.disease_ids == other.disease_ids):
            return False

        if {hpo.id for hpo in self.hpos} != {hpo.id for hpo in other.hpos}:
            return False

        return True


@dataclass
class Patient:
    id: PatientId
    diseases: Dict[str, Disease]

    def to_json(self) -> Dict[PatientId, Patient]:
        dict_obj = self.__dict__.copy()
        dict_obj["diseases"] = [
            disease_obj.to_json() for diseas_id, disease_obj in dict_obj["diseases"].items()
        ]
        return dict_obj


@dataclass
class PatientSymptoms:
    """증상유사도 API의 클라이언트 전달받는 요청의 데이터클레스"""

    symptoms: list

    def __post_init__(self):
        self.symptoms = {Phenotype(phenotype) for phenotype in self.symptoms}


@dataclass
class APIResponse:
    """증상유사도 API의 클라이언트 전달되는 응답의 데이터클레스"""

    diseaes_similarity: Dict[str, float]


PatientId = NewType("PatientId", str)
