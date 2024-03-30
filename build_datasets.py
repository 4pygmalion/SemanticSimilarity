from __future__ import annotations

import os
import sys
import io
import gc
import json
import logging
import random

import pandas as pd
import pronto
import requests

from pronto import Ontology
from typing import Set, List, Dict, FrozenSet, Union

SEMANTIC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SEMANTIC_DIR)
sys.path.append(SEMANTIC_DIR)
from SemanticSimilarity.data_model import Phenotype, Disease, PatientId, Patient


class PhenotypeIdNotInOntologyException(Exception):
    pass


class TypeMissmatchException(Exception):
    pass


class EmptyPatientsException(Exception):
    pass


class OntologyHandler:
    """
    obo파일, hpoa파일을 읽어서 테스트를 위한 데이터셋을 만드는 클래스
    """

    def __init__(self, config: dict, logger: logging.Logger = logging.Logger(__name__)) -> None:
        """

        Args:
            - obo_file_path: file path to hpo ontology
            - dataframe_path: file path to disease-phenotype relation
            - usecols: columns to use on disease-phenotpye relation dataframe

        Note:
            >>> phenotype.hpoa.columns
            >>> ['#DatabaseID', 'DiseaseName', 'Qualifier', 'HPO_ID', 'Reference',
                'Evidence', 'Onset', 'Frequency', 'Sex', 'Modifier', 'Aspect',
                'Biocuration'
                ]
        """
        self.config = config
        self.logger = logger
        self.set_ontology()
        self.set_disease_pheno_map()

    def set_ontology(self) -> None:
        self.logger.info("Read ontology.obo file")
        obo_path = os.path.join(ROOT_DIR, self.config["FILE"]["HPO_OBO_DISK"])
        if os.path.exists(obo_path):
            self.logger.debug(f"Read ontology file from disk: {obo_path}")
            self.ontology = Ontology(obo_path)
            return

        msg = f"Read ontology file from github: {self.config['FILE']['HPO_OBO']}"
        self.logger.debug(msg)
        response = requests.get(self.config["FILE"]["HPO_OBO"])
        self.ontology = Ontology(io.BytesIO(response.content))

        return

    def set_disease_pheno_map(self) -> None:
        self.logger.info("Build disease-phenotype mapping dict")

        disease_pheno_df = pd.read_csv(
            io.StringIO(requests.get(self.config["FILE"]["DISEASE_HPO"]).text),
            skiprows=4,
            sep="\t",
            usecols=self.config["COLS"]["DISEASE_HPO"],
        )
        self.disease_pheno_map = self._build_disease_pheno_map(disease_pheno_df=disease_pheno_df)
        self.all_phenos = set(self.ontology.keys())

        del disease_pheno_df
        gc.collect()

        return

    def _build_disease_pheno_map(self, disease_pheno_df: pd.DataFrame) -> Dict[str, Phenotype]:
        """
        disease id로 접근할 수 있는 disease-phenotypes dictionary를 구성함.

        Examples:
            >>> diseases = self._build_disease_pheno_map(disease_pheno_df)
            >>> diseases = {
                    "OMIM:619340": Disease(
                        disease_name="Developmental and epileptic encephalopathy 96",
                        disease_id="OMIM:619340",
                        pheno={
                            Phenotype(name="Epileptic spasm", hpoid="HP:0011097"),
                            Phenotype(
                                name="Intellectual disability, profound",
                                hpoid="HP:0002187",
                            ),
                        },
                    ),
                    "OMIM:619341": Disease(
                        disease_name="Developmental and epileptic encephalopathy 100",
                        disease_id="OMIM:619341",
                        pheno={
                            Phenotype(name="Tonic seizure", hpoid="HP:0032792"),
                            Phenotype(
                                name="Small for gestational age", hpoid="HP:0001518"
                            ),
                        },
                    ),
                }
        """
        disease_pheno_map = dict()

        disease_pheno_records = (
            disease_pheno_df.groupby(["#DatabaseID", "DiseaseName"])["HPO_ID"]
            .apply(set)
            .reset_index(name="hpo_ids")
        ).to_dict("records")

        for record in disease_pheno_records:
            phenos = set()
            for hpoid in list(record["hpo_ids"]):
                try:
                    phenos.add(self.get_phenotype_from_hpoid(hpoid))
                except PhenotypeIdNotInOntologyException:
                    continue

            disease_pheno_map[record["#DatabaseID"]] = Disease(
                disease_id=record["#DatabaseID"],
                disease_name=record["DiseaseName"],
                pheno=phenos,
            )
        return disease_pheno_map

    def get_phenotype_from_hpoid(self, hpoid: str) -> Phenotype:
        """
        get Phenotype data given hpoid

        Raises:
            ValueError: given hpoid is NOT in the ontology
        Examples:
            >>> self.get_phenotype_from_hpoid("HP:0011097")
            >>> Phenotype(id="HP:0011097", name="Epileptic spasm")
        """
        if hpoid not in self.ontology.keys():
            raise PhenotypeIdNotInOntologyException(f"{hpoid} is NOT in the ontology")

        return Phenotype(name=self.ontology[hpoid].name, id=self.ontology[hpoid].id)

    def build_optimal_patients_map(
        self, disease_ids: List[str] = None, max_pheno: int = 20
    ) -> Dict[str, Disease]:
        """
        disease_pheno에서 정의되는 disease-pheno 관계에서(Disease:Phenotypes = 1:n)
        해당 질병의 phenotype을 임의로 추출.

        Note:
            - disease_ids가 주어지면 해당 질병에 대해서만 optimal_patients_map 구축.
            - disease_ids가 주어지지 않으면 self.disease_pheno_map의 모든 질병
            (=self.disease_pheno_map.keys())에 대해 optimal_patients_map 구축

        Args:
            disease_ids (List[str]): 질병 id 리스트
            max_pheno (int): 1 질병 당 최대 증상 수

        Returns:
            Dict[str, Disease]: key, value = 질병 id, Disease
        """
        disease_to_resolve = disease_ids if disease_ids else set(self.disease_pheno_map.keys())

        optiaml_patients_map = dict()
        for disease_id in disease_to_resolve:
            disease = self.disease_pheno_map[disease_id]
            selected_phenos = disease.get_n_phenos_randomly(max_pheno)

            optiaml_patients_map[disease_id] = Disease(
                disease_name=disease.disease_name,
                disease_id=disease.disease_id,
                pheno=selected_phenos,
            )
        return optiaml_patients_map

    def build_noisy_patients_map(
        self,
        disease_ids: List[str] = None,
        noise_ratio: float = 0.2,
    ) -> Dict[str, Disease]:
        """
        Note:
            disease_pheno에서 정의되는 disease-pheno 관계에서(Disease:Phenotypes = 1:n)
            해당 질병의 phenotype을 임의로 추출한 후, 해당 disease와 관계 없는 phenotype를 추가.

            noise_ratio는 추가되는 noise-pheno의 비율.
            가령, 해당 질병의 전체 phenotype 수를 k라 할 때, noise_ratio=0.2 이면
            noise-pheno의 개수는 int(0.2*k) 개.

            num_noise의 값이 0이면 이미 해당 질병은 phenotype의 수가 적은 것이므로,
            noise를 추가하지 않음.

        Args:
            disease_ids (List[str]): 질병 id 리스트
            noise_ratio (float): 해당 질병과 상관 없는 phenotype의 비율
            max_pheno (int): 1 질병 당 최대 증상 수

        Returns:
            Dict[str, Disease]: key, value = 질병 id, Disease
        """
        # build optimal patients in advance.
        optimal_patients: Dict[str, Disease] = self.build_optimal_patients_map()

        disease_to_resolve = disease_ids if disease_ids else set(optimal_patients.keys())
        nosiy_patients_map = dict()

        for disease_id in disease_to_resolve:
            optimal_pheno = optimal_patients[disease_id].pheno
            num_noise = int(len(optimal_pheno) * noise_ratio)
            selected_noise_phenos = set()
            if num_noise > 0:
                subtraction = self.all_phenos - self.get_super_pheno_from(
                    hpo_termset={self.ontology[pheno.id] for pheno in optimal_pheno},
                    with_self=True,
                )
                noise_hpos = random.sample(subtraction, num_noise)

                for i in range(num_noise):
                    optimal_pheno.pop()

                selected_noise_phenos = set()
                for hpoid in noise_hpos:
                    try:
                        selected_noise_phenos.add(self.get_phenotype_from_hpoid(hpoid))
                    except ValueError:
                        pass

            nosiy_patients_map[disease_id] = Disease(
                disease_name=optimal_patients[disease_id].disease_name,
                disease_id=disease_id,
                pheno=selected_noise_phenos | optimal_pheno,
            )

        return nosiy_patients_map

    def build_ambiguous_patient_map(
        self,
        disease_ids: List[str] = None,
        ambiguous_ratio: float = 0.2,
    ) -> Dict[str, Disease]:
        """
        disease_pheno에서 정의되는 disease-pheno 관계에서(Disease:Phenotypes = 1:n)
        해당 질병의 phenotype을 임의로 추출한 후, 해당 disease의 phenotype 중 일부를
        그 phenotype의 조상으로 대체.

        Args:
            disease_ids (List[str]): 질병 id 리스트
            noise_ratio (float): 조상으로 대체될 phenotype의 비율
            max_pheno (int): 1 질병 당 최대 증상 수

        Returns:
            Dict[str, Disease]: key, value = 질병 id, Disease
        """
        # build optimal patients in advance.
        optimal_patients: Dict[str, Disease] = self.build_optimal_patients_map()
        ambiguous_patients_map = dict()

        disease_to_resolve = disease_ids if disease_ids else set(optimal_patients.keys())

        for disease_id in disease_to_resolve:
            optimal_pheno = optimal_patients[disease_id].pheno
            ancestors = self.get_super_pheno_from(
                hpo_termset={self.ontology[pheno.id] for pheno in optimal_pheno}
            )

            num_ambiguous = int(len(optimal_pheno) * ambiguous_ratio)
            ancestors_selected = set()

            if num_ambiguous > 0:
                for i in range(num_ambiguous):
                    optimal_pheno.pop()

                sampled = random.sample(ancestors, num_ambiguous)
                ancestors_selected = set()
                for hpoid in sampled:
                    try:
                        ancestors_selected.add(self.get_phenotype_from_hpoid(hpoid))
                    except ValueError:
                        pass

            ambiguous_patients_map[disease_id] = Disease(
                disease_name=optimal_patients[disease_id].disease_name,
                disease_id=disease_id,
                pheno=optimal_pheno | ancestors_selected,
            )

        return ambiguous_patients_map

    def get_super_pheno_from(
        self, hpo_termset: Set[pronto.term.Term], with_self: bool = False
    ) -> FrozenSet[str]:
        """
        hpo_termset이 주어질 때, 해당 hpoid들의 공통조상을 모두 반환(합집합).
        단, 자기 자신은 제외.

        Args:
            hpo_termset (Set[pronto.term.Term]): set of pronto.term.Term

        Returns:
            FrozenSet[str]: set of super-hpoids
        """
        if type(hpo_termset) != set:
            raise TypeMissmatchException

        term_set = pronto.TermSet(hpo_termset)
        return term_set.superclasses(with_self=with_self).to_set().ids

    def get_avg_pheno_num(self, patients: Dict[str, Disease]) -> float:
        """
        가상 환자(=질병을 1개만 가지는 환자)의 평균 증상 수를 계산
        Returns:
            float: 평균 증상 수
        Examples:
            >>> self.get_avg_pheno_num(patients)
            >>> 7.5
        """
        if len(patients) == 0:
            raise EmptyPatientsException

        num_phenos = 0
        for pat_id, pat_obj in patients.items():
            num_phenos += len(pat_obj.pheno)
        return num_phenos / len(patients)

    def get_avg_pheno_num_real_world(self, patients: Dict[PatientId, Patient]) -> float:
        """
        각 환자의 평균 증상 수를 계산
        Args:
            patients: patientId로 특정되는 환자 딕셔너리
        Returns:
            float:평균 증상 수
        Examples:
            >>> self.get_avg_pheno_num(patients)
            >>> 4.5
        """
        if len(patients) == 0:
            raise EmptyPatientsException

        num_phenos = 0
        for pat_id, pat_obj in patients.items():
            disease_id, disease_obj = list(pat_obj.diseases.items())[0]
            num_phenos += len(disease_obj.pheno)
        return num_phenos / len(patients)

    def save_to_json(
        self,
        source_dict: Union[Dict[str, Disease], Dict[PatientId, Patient]],
        target_path: str,
    ) -> None:
        target_dict = {}
        for key, disease in source_dict.items():
            target_dict[key] = disease.to_json()

        with open(target_path, "w") as fp:
            json.dump(target_dict, fp)

    def load_from_json(self, json_path: str) -> Dict[str, Disease]:
        """
        self.build_optimal_patients_map() 또는 self.build_noisy_patients_map()으로
        만들어진 disease-phenotype mapping이 .json 파일로 기록되어 있는 경우
        해당 json 파일을 읽어서 dictionary mapping을 복원하는 함수

        Args:
            json_path (str): path to disease-phenotype mapping .json

        Returns:
            Dict[str, Disease]: key, value = 질병 id, Disease

        """
        disease_pheno_map = dict()
        with open(json_path, "r") as fp:
            raw_dict = json.load(fp)

        for key, val in raw_dict.items():
            phenos = set(Phenotype(name=pheno["name"], id=pheno["id"]) for pheno in val["pheno"])
            disease_pheno_map[key] = Disease(
                disease_id=val["disease_id"],
                disease_name=val["disease_name"],
                pheno=phenos,
            )
        return disease_pheno_map

    def load_real_world_case(
        self, json_path: str, n_patients: int = 100
    ) -> Dict[PatientId, Patient]:
        """
        실제 진단된 환자 데이터에서 평가를 위한 dictionary mapping을 추출하고 반환.
        전체 환자 중 n_patients 명을 추출.

        Examples:
            >>> real_world_raw_file = {
                "EPJ21-PSFY": {
                    "OMIM:619340": [
                        "HP:0000559",
                        "HP:0100689",
                        "HP:0000615",
                        "HP:0000613",
                        "HP:0000518",
                    ]
                },
                "GTX22-WWBY": {
                    "OMIM:163950": [
                        "HP:0003502"
                    ]
                }
                ..
            >>> self.load_real_world_case(json_path, n_patients)
            >>> {
                    "EPJ21-PSFY": Patient(
                        id="EPJ21-PSFY",
                        diseases={
                            "OMIM:619340": Disease(
                                disease_id="OMIM:619340",
                                disease_name="DEVELOPMENTAL AND EPILEPTIC ENCEPHALOPATHY 96",
                                pheno={
                                    Phenotype(id="HP:0000559"),
                                    Phenotype(id="HP:0100689"),
                                    Phenotype(id="HP:0000615"),
                                    Phenotype(id="HP:0000613"),
                                    Phenotype(id="HP:0000518"),
                                },
                            ),
                            "OMIM:619341": Disease(
                                disease_id="OMIM:619341",
                                disease_name="PRAJA RING FINGER UBIQUITIN LIGASE 2",
                                pheno={Phenotype(id="HP:0006554")},
                            ),
                        },
                    ),
                    ...
                }
        """

        with open(json_path, "r") as fp:
            patients_dict: Dict[str, Dict[str, str]] = json.load(fp)

        actual_patients = dict()
        patients_list = list(patients_dict.items())

        loop_counter = 0
        while (len(actual_patients) < n_patients) and (loop_counter < n_patients):
            patient_id, patient_diseases = (
                random.choice(patients_list)
                if n_patients < len(patients_list)
                else patients_list[loop_counter]
            )

            is_valid_disease = set(patient_diseases.keys()).issubset(
                set(self.disease_pheno_map.keys())
            )
            for patient_disease_id in patient_diseases.keys():
                if is_valid_disease:
                    phenos = set(
                        Phenotype(id=pheno) for pheno in patient_diseases[patient_disease_id]
                    )

                    disease = Disease(
                        disease_id=patient_disease_id,
                        disease_name=self.disease_pheno_map[patient_disease_id].disease_name,
                        pheno=phenos,
                    )

                    if patient_id in actual_patients.keys():
                        actual_patients[patient_id].diseases[disease.disease_id] = disease
                    else:
                        actual_patients[patient_id] = Patient(
                            id=patient_id,
                            diseases={disease.disease_id: disease},
                        )
            loop_counter += 1

        return actual_patients
