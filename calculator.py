import os
import io
import sys
import math
import logging
import requests

from typing import Dict, Set, Tuple, List
from collections import defaultdict

import pronto
import numpy as np
import pandas as pd
from scipy.special import softmax

SEMANTIC_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SEMANTIC_SIM_DIR)
sys.path.append(ROOT_DIR)
from SemanticSimilarity.data_model import Phenotype, Gene

FRQUENCY_AVG = {
    "HP:0040280": 1.0,  # Obligate
    "HP:0040281": 0.895,  # Very frequent
    "HP:0040282": 0.545,  # Frequent
    "HP:0040283": 0.17,  # Ocausal
    "HP:0040284": 0.025,  # Very rare
    "HP:0040285": 0,
}


def cal_symtpom_similarity_from_lambda(hpos: List[str], timeout: int = 100) -> Dict[str, float]:
    """외부 AWS Lambda 서비스를 통해 증상 유사도 점수를 계산

    이 함수는 주어진 HPO(Human Phenotype Ontology) 용어 목록을 기반으로 증상 유사도 점수를 계산하기 위해
    AWS Lambda 서비스에 POST 요청. Lambda 서비스는 계산을 수행하고 질병에 대한 유사도 점수를 반환

    Note:
        Lambda 비용: 다음 150억GB-초/월	GB-초당 0.0000133334 USD
        https://calculator.aws/#/addService/Lambda

    Args:
        hpos (List[str]): 증상 유사도 점수를 계산할 HPO 용어의 목록입니다.
        timeout (int, optional): HTTP 요청의 제한 시간(초)입니다. 기본값은 100초

    Returns:
        Dict[str, float]: 질병 식별자를 키로 하고 증상 유사도 점수(0에서 1 사이의 부동소수점 값)를 값으로 갖는
                          딕셔너리를 반환

    Raises:
        requests.HTTPError: HTTP 요청 또는 응답 처리 중에 오류가 발생한 경우 예외가 발생

    Example:
        hpos = ['HP:0000198', 'HP:0000564']
        similarity_scores = cal_symtpom_similarity_from_lambda(hpos, timeout=120)
        print(similarity_scores)
    """

    query_hpo = "\n".join(hpos)

    query = f"START\nREQUEST:DISEASE\n{query_hpo}\nEND"
    try:
        response = requests.post(
            url="https://dekc2xej2l.execute-api.ap-northeast-2.amazonaws.com/v1/hposim",
            headers={"Content-Type": "application/json"},
            json={"request": query},
            timeout=timeout,
        )
        response.raise_for_status()
    except:
        raise requests.HTTPError("symptom similarity request error")

    symptom_similarity_scores = dict()
    for line in response.json()["body"]:
        row = line.split(",")
        if len(row) != 3:
            continue

        disease_id, score, _ = row
        if disease_id.startswith("PMID:"):
            disease_id = disease_id.replace("PMID:", "3B:")

        elif disease_id.isdigit():
            disease_id = "OMIM:" + disease_id

        symptom_similarity_scores[disease_id] = (
            float(score) if all(d.isdigit() for d in score.split(".")) else 0
        )

    return symptom_similarity_scores


class BaseSimilarityCalculator:
    def __init__(
        self,
        config: dict,
        hpo_ontology: pronto.ontology = None,
        logger=logging.Logger(__name__),
    ) -> None:
        self.config = config
        self.hpo_ontology = hpo_ontology
        if not hpo_ontology:
            self.hpo_ontology = pronto.Ontology(
                io.BytesIO(requests.get(config["FILE"]["HPO_OBO"]).content)
            )

        self.logger = logger
        self._precalculated()

    def _precalculated(self) -> None:
        self.n_ontology = len(self.hpo_ontology)

        self.disease_pheno_map = dict()
        self.hpo2depth = dict()
        self.subclass_map: Dict[str, Set[str]] = self.build_subclass_map()
        self.n_subclass = {
            pheno_id: len(subclasses) for pheno_id, subclasses in self.subclass_map.items()
        }
        self.ic = {
            pheno_id: -math.log(n_subclass / self.n_ontology) if n_subclass != 0 else 0
            for pheno_id, n_subclass in self.n_subclass.items()
        }

    def build_subclass_map(self) -> dict:
        """
        pronto.Term.subclasses()의 시간복잡도가 O(n) where n= #of subclasses
        given node 이므로, 모든 hpo term에 대한 subclass를 미리 계산해서 멤버변수
        self.subclass_map에 저장
        """
        subclass_map = dict()
        for term in self.hpo_ontology.terms():
            subclass_map[term.id] = set(term.subclasses(with_self=True).to_set().ids)

        return subclass_map

    def update_subclass_level(self, phenotype: Phenotype, phenotype2level: Dict[str, int]) -> None:
        """주어진 Phenotype에 대해서, 바로 아래의 자손의 Phenotype의 레벨을
        Phenotype2level에 업데이트함

        Args:
            phenotype (Phenotype): 표현형 개념
            phenotype2level (Dict[str, int]): phenotype-level 매핑딕셔너리

        """
        term = self.hpo_ontology[phenotype.id]
        level = max(phenotype2level[phenotype.id])
        for subterm_id in term.subclasses(distance=1, with_self=False).to_set().ids:
            phenotype2level[subterm_id].append(level + 1)
            self.update_subclass_level(Phenotype(subterm_id), phenotype2level)

        return

    def set_disease_pheno_map(self, disease_pheno_map: Dict[str, Set[Phenotype]]) -> None:
        self.disease_pheno_map = disease_pheno_map

        return

    def set_level(self) -> None:
        """HPO의 모든 개념에 대해서 HPO:level 매핑 딕셔너리를 생성하여 인스턴스변수에
        저장.

        Example:
            >>> self.set_level()
            >>> print(self.phenotype2level["HP:0000001"])
            1
        """

        self.logger.info("Build HPO2level dictionary.")

        root_hpo_id = "HP:0000001"
        phenotype2level = defaultdict(list)
        phenotype2level[root_hpo_id].append(1)

        self.update_subclass_level(Phenotype(root_hpo_id), phenotype2level)

        res = defaultdict(int)
        for phenotype_id, levels in phenotype2level.items():
            res[phenotype_id] = max(levels)

        self.phenotype2level = res
        return

    def get_level(self, phenotype: Phenotype) -> int:
        return self.phenotype2level[phenotype.id]

    def set_mica_mat(self) -> None:
        """MICA matrix을 인스턴스 변수에 저장

        Note:
            >>> np.zeros(shape=(10, 10), dtype=np.unicode_)
            array([['', '', '', '', '', '', '', '', '', ''],
                ['', '', '', '', '', '', '', '', '', ''],
                ['', '', '', '', '', '', '', '', '', ''],
                ['', '', '', '', '', '', '', '', '', ''],
                ['', '', '', '', '', '', '', '', '', ''],
                ['', '', '', '', '', '', '', '', '', ''],
                ['', '', '', '', '', '', '', '', '', ''],
                ['', '', '', '', '', '', '', '', '', ''],
                ['', '', '', '', '', '', '', '', '', ''],
                ['', '', '', '', '', '', '', '', '', '']], dtype='<U1')
        """
        self.logger.info("Build MICA matrix.")

        phenotype_id_ascending_level: list = sorted(
            self.phenotype2level, key=self.phenotype2level.get
        )

        hpoid2idx = {hpo_id: idx for idx, hpo_id in enumerate(phenotype_id_ascending_level)}

        mica_mat = np.zeros(
            shape=(
                len(phenotype_id_ascending_level),
                len(phenotype_id_ascending_level),
            ),
            dtype="<U10",
        )
        for hpo_idx, hpo_id in enumerate(phenotype_id_ascending_level):
            if not hpo_id.startswith("HP:"):
                continue

            # expression below is equivalent to "with_self=True"
            self.subclass_map[hpo_id].add(hpo_id)
            subterm_ids = self.subclass_map[hpo_id]

            for subterm_id in subterm_ids:
                subterm_idx = hpoid2idx[subterm_id]
                mica_mat[hpo_idx, subterm_idx] = hpo_id

                for subterm_id2 in subterm_ids:
                    subterm_idx2 = hpoid2idx[subterm_id2]
                    mica_mat[subterm_idx, subterm_idx2] = hpo_id

        self.hpoid2idx = hpoid2idx
        self.mica_mat = mica_mat
        return

    def get_most_informative_common_ancestor(
        self, phenotype1: Phenotype, phenotype2: Phenotype
    ) -> Phenotype:
        """두 개념의 공통조상 중 가장 정보력이 높은(MICA)의 개념을 불러옴

        Args:
            phenotype1 (Phenotype): Phenotype내 개념에 해당하는 Phenotype ID
            phenotype2 (Phenotype): Phenotype내 개념에 해당하는 Phenotype ID

        Returns:
            (Phenotype): MICA에 해당하는 Phenotype ID
        """

        return Phenotype(
            self.mica_mat[self.hpoid2idx[phenotype1.id], self.hpoid2idx[phenotype2.id]]
        )

    def get_information_content(self, phenotype: Phenotype) -> float:
        """개념(phenotype)의 IC값을 계산하여 반환

        Note:
            RelativeBestPair method
            # p(x) = |I(x)| / |I(T)|
                , where I(x) 하위 개념을 포함한 개수, |I(T)|: 전체 개념의 수
            # IC(x) = -log(p(x))
            reference: https://url.kr/bnyshe

        Args:
            phenotype (Phenotype): Phenotype ID

        Return
            information content(IC) (float): [0, inf)사이의 값을 가진 IC

        """
        return self.ic[phenotype.id]

    def get_semantic_similarity_one_side(
        self, phenotypes1: Set[Phenotype], phenotypes2: Set[Phenotype]
    ):
        """두 집합 개념(phenotypes, phenotypes)의 ontology의 유사성을 단방향
        Q(phenotypes, 쿼리)-> D(phenotypes, 질환)으로 유사성을 계산함

        Note:
            sim(Q->D): Avg[Sum_{t in Q} (max t2 in D IC(MICA(t1, t2)))]
            (https://user-images.githubusercontent.com/86948539/225186395-bbc775c6-1df4-4185-b118-fd37e188fa19.png)

        Args:
            phenotypes1 (Set[Phenotype]): Phenotype 개념의 집합
            phenotypes2 (Set[Phenotype]): Phenotype 개념의 집합

        Raises:
            NotImplementedError: 서브클레싱하여 만든 클레스에서 해당 함수가 정의되지
                않은 경우
        """
        raise NotImplementedError

    def get_semantic_similarity(
        self, phenotypes1: Set[Phenotype], phenotypes2: Set[Phenotype]
    ) -> float:
        """두 집합개념 (concept1, concept2)의 ontology에서 의미론적 유사도를 계산함.

        self.get_semantic_similarity_one_side을 호출하여, 양방향의 의미론적 유사도를
        계산함.

        Args:
            phenotypes1 (Set[Phenotype]): Phenotype 개념의 집합
            phenotypes2 (Set[Phenotype]): Phenotype 개념의 집합

        Raises:
            NotImplementedError: 서브클레싱하여 만든 클레스에서 해당 함수가 정의되지
                않은 경우
        """

        raise NotImplementedError

    def get_disease_similarty(self, patient_hpos: Set[Phenotype]) -> Dict[str, float]:
        """환자의 Phenotype의 집합 주어졌을 때의, Phenotype에 등재된 질환들 사이의 질병유사도를
        계산하여 반환

        Note:
            유사도 점수의 범위는 [0, inf).

        Args:
            patient_hpos (set): 환자의 Phenotype의 집합

        Returns:
            dict: 질환별 환자증상-질환증상의 유사도

        Example:
            >>> self.get_disease_similarity(patient_hpos={....})
            {
                "OMIM:619344": 0.58,
                "OMIM:619345": 0.48,
                "OMIM:619346": 5.38,
                "OMIM:619347": 0.28,
                "OMIM:619348": 0.18,
                "OMIM:619349": 0.08,
                "OMIM:619340": 1.99,
                "OMIM:619350": 0.88,
                "OMIM:619341": 0.78,
                "OMIM:619343": 0.68,
            }
        """
        disease_similarity: Dict[str, float] = dict()
        for disease_id, disease_obj in self.disease_pheno_map.items():
            disease_similarity[disease_id] = self.get_semantic_similarity(
                disease_obj.pheno, patient_hpos
            )

        return disease_similarity


class ResnikSimilarityCalculator(BaseSimilarityCalculator):
    def __init__(self, config, hpo_ontology=None, logger=logging.Logger(__name__)) -> None:
        super().__init__(config, hpo_ontology, logger)

    def get_semantic_similarity_one_side(
        self, phenotypes1: Set[Phenotype], phenotypes2: Set[Phenotype]
    ) -> float:
        """두 개념집합의 의미론적 유사도를 Concepts 1-> Concept2에 대해서 구함

        Note:
            sim(Q->D): Avg[Sum_{t in Q} (max t2 in D IC(MICA(t1, t2)))]
            (https://user-images.githubusercontent.com/86948539/225186395-bbc775c6-1df4-4185-b118-fd37e188fa19.png)

        Args:
            phenotypes1 (Set[Phenotype]): 개념의 집합
            phenotypes2 (Set[Phenotype]): 개념의 집합

        Returns:
            float: 의미론적 유사도(Semantic similarity)
        """
        information_contents = list()
        for phenotype1 in phenotypes1:
            ics = list()
            for phenotype2 in phenotypes2:
                try:
                    mica = self.get_most_informative_common_ancestor(phenotype1, phenotype2)
                    ic = self.get_information_content(mica)
                except KeyError:  # edge case: hpoid not in self.hpo2idx
                    ic = 0

                ics.append(ic)
            max_information_content = max(ics)

            information_contents.append(max_information_content)

        return sum(information_contents) / len(information_contents)

    def get_semantic_similarity(self, concepts1: Set[Phenotype], concepts2: Set[Phenotype]) -> dict:
        semantic_similarity = (
            self.get_semantic_similarity_one_side(concepts1, concepts2)
            + self.get_semantic_similarity_one_side(concepts2, concepts1)
        ) / 2

        return semantic_similarity


class NodeLevelSimilarityCalculator(BaseSimilarityCalculator):
    """MICA(Most informatics common ancestor)의 계층레벨을 IC대신에 사용중

    Note:
        GEBRA에서 사용중인 계산방식임
    """

    def __init__(
        self,
        config: dict,
        hpo_ontology: pronto.ontology.Ontology = None,
        logger=logging.Logger(__name__),
    ) -> None:
        super().__init__(config, hpo_ontology, logger)

    def get_semantic_similarity_one_side(
        self, concepts1: Set[Phenotype], concpets2: Set[Phenotype]
    ) -> float:
        """두 개념집합의 의미론적 유사도를 Concepts 1-> Concept2에 대해서 구함

        Note:
            sim(Q->D): Avg[Sum_{t in Q} (max t2 in D Level(MICA(t1, t2)))]
            - MICA의 IC 대신에 level(=depth)을 기준으로 계산함.

        Args:
            concepts1 (Set[Phenotype]): 개념의 집합
            concpets2 (Set[Phenotype]): 개념의 집합

        Returns:
            float: 의미론적 유사도(Semantic similarity)
        """

        mica_depths = list()
        for concept1 in concepts1:
            depths = list()
            for concep2 in concpets2:
                try:
                    mica = self.get_most_informative_common_ancestor(concept1, concep2)
                    depth = self.get_level(mica)
                except:
                    depth = 0

                depths.append(depth)

            max_depth = max(depths)
            mica_depths.append(max_depth)

        return sum(mica_depths) / len(mica_depths)

    def get_semantic_similarity(self, concepts1: Set[Phenotype], concepts2: Set[Phenotype]) -> dict:
        semantic_similarity = (
            self.get_semantic_similarity_one_side(concepts1, concepts2)
            + self.get_semantic_similarity_one_side(concepts2, concepts1)
        ) / 2

        return semantic_similarity


class FrequentBasedSimilarityCaculator(ResnikSimilarityCalculator):
    def __init__(
        self,
        config: dict,
        hpo_ontology: pronto.ontology.Ontology = None,
        logger=logging.Logger(__name__),
    ) -> None:
        super().__init__(config, hpo_ontology, logger)

    def set_disease_hpo(self, cutoff: int = 20) -> None:
        """질병에 따른 증상(Phenotype)의 빈도를 인스턴스 변수에 저장

        Note:
            질병의 증상의 빈도가 없는 경우 default=0을 할당

        Args:
            cutoff (int): 질환을 가진 샘플의 최소 N수
        """
        df = pd.read_csv(
            self.config["FILE"]["DISEASE_HPO"],
            skiprows=4,
            sep="\t",
            usecols=self.config["COLS"]["DISEASE_HPO"],
        )

        disease_hpos = defaultdict(dict)
        for idx, row in df.iterrows():
            frequency = row["Frequency"]
            if frequency in FRQUENCY_AVG:
                # frequency = FRQUENCY_AVG[frequency]
                frequency = 0.0
            elif isinstance(frequency, str):
                if frequency.endswith("%"):
                    frequency = float(frequency.rstrip("%")) * 0.01  # ex) "50%"
                else:
                    numerater, denominator = frequency.split("/")
                    frequency = eval(frequency) if int(denominator) >= cutoff else 0  # ex) "1/2"

            elif math.isnan(frequency):
                frequency = 0.0

            disease_hpos[row["#DatabaseID"]][row["HPO_ID"]] = frequency

        self.disease_hpos = disease_hpos

        return

    def get_frequency_semantic_similarity_one_side(
        self, patient_hpos: Set[Phenotype], disease_id: str
    ) -> float:
        """두 개념집합의 의미론적 유사도를 Concepts 1-> Concept2에 대해서 구함

        Note:
            sim(Q->D): Avg[Sum_{t in Q} (max t2 in D (1+freq)*IC(MICA(t1, t2)))]
            각 IC을 계산한 다음에, (1 + freq)을 연산해주도록함.
            - why? default 0 값보다는 freq가 있는 경우를 up-weigting해주기위함

        Args:
            concepts1 (Set[Phenotype]): 개념의 집합
            concpets2 (Set[Phenotype]): 개념의 집합

        Returns:
            float: 의미론적 유사도(Semantic similarity)
        """
        mica_ic_list = list()
        for patient_hpo in patient_hpos:
            ic_freq = []
            sum_freq = sum(self.disease_hpos[disease_id].values())

            # self.disease_hpos := Dict[HpoStr, float]
            for disease_hpo, freq in self.disease_hpos[disease_id].items():
                try:
                    mica = self.get_most_informative_common_ancestor(
                        patient_hpo, Phenotype(disease_hpo)
                    )
                    ic = self.get_information_content(mica)
                    normalized_freq = freq / sum_freq if sum_freq != 0 else freq
                    ic_freq.append(normalized_freq + ic)
                except:
                    ic = 0
                    ic_freq.append(ic)

            mica_ic_freq = max(ic_freq)  # mica_ic_freq = sum(mica_ic_freq))
            mica_ic_list.append(mica_ic_freq)
        return sum(mica_ic_list) / len(mica_ic_list)

    def get_disease_similarty(self, patient_hpos: Set[Phenotype]) -> dict:
        """Frequency based 질병유사도 계산

        Args:
            patient_hpos (Set[Phenotype]): 환자의 Phenotype의 집합

        Returns:
            dict: 질환별 질병유사도 점수
        """
        disease_similarity = dict()
        for disease_id, hpo_freqs in self.disease_hpos.items():
            patient_to_disease_similarity = self.get_frequency_semantic_similarity_one_side(
                patient_hpos=patient_hpos, disease_id=disease_id
            )

            disease_to_patient_similarity = self.get_semantic_similarity_one_side(
                set(Phenotype(hpoid) for hpoid in hpo_freqs.keys()),
                patient_hpos,
            )
            similarity = (patient_to_disease_similarity + disease_to_patient_similarity) / 2

            disease_similarity[disease_id] = similarity

        return disease_similarity


class GeneDiseaseSimilarityCalculator(ResnikSimilarityCalculator):
    """유전자(Gene)자 내에 존재하는 Phenotype들 중에 가장 유사한 유전자를 찾고,
    그 유전자의 질환들은 Upweighting하여 가중치를 활용하는 전략

    Note:
        예시) F13A1유전자는 응고인자(Coagulation factor XIII A)에 관련된 유전자인데,
        이 유전자의 연결된 HPO을 보면, 대 다수가 Bleeding과 관련된 유전자이다.

        - HP:0000225 (Gingival bleeding)
        - HP:0002170 (Intracranial hemorrage)
        - HP:0000421 (Epstaxis)

        Similarity =
            질환1 = Sim(환지HPO -> 질환HPO) + Sim(질환HPO -> 환자HPO)
            질환2 = Sim(환지HPO -> 질환HPO) + Sim(질환HPO -> 환자HPO)
            ..
            질환N = Sim(환지HPO -> 질환HPO) + Sim(질환HPO -> 환자HPO)

        유전자 레벨 표현형 유사성 =
            유전자1: Sim(환자HPO -> 유전자HPO) + Sim(유전자HPO -> 환자HPO)
            ...
            유전자N: Sim(환자HPO -> 유전자HPO) + Sim(유전자HPO -> 환자HPO)

        유전자 표현형 유사성 S= (s1, s2... sn)
        정규화된 유전자 표현형 유사성 (1 + softmax(S))

        정규화된 유전자 유사성에 따라, 유전자내 질환들에게 up-weighting하여 반환


    Example:
        >>> calculator = GeneDiseaseSimilarityCalculator(
                config=OmegaConf.load("...")
            )
        >>> calculator.set_gene_info()
        >>> calculator.set_level()
        >>> calculator.set_mica_mat()
        >>> calculator.get_gene_patient_symtom_similarity(
                {Phenotype("HP:0000003")}
            )
    """

    def __init__(
        self,
        config,
        hpo_ontology: pronto.ontology.Ontology = None,
        logger=logging.Logger(__name__),
    ) -> None:
        super().__init__(config, hpo_ontology, logger)
        self.gene_info = dict()

    def set_gene_info(self) -> None:
        """Gene 정보를 인스턴스 변수 내에 저장.

        Note:
            genes_to_hpoenotype.txt의 파일 포맷은 아래와 같음.
            #Format: entrez-gene-id<tab>...<tab>disease-ID for link
            8192    CLPP    HP:0000013    Hypoplasia of the uterus       ...
            8192    CLPP    HP:0000815    Hypergonadotropic hypogonadism ...
            8192    CLPP    HP:0000786    Primary amenorrhea      -      ...
            8192    CLPP    HP:0000007    Autosomal recessive inheritance...

        Example:
            >>> self.set_gene_info()
            >>> print(self.gene_info)
            {
                "8192": Gene(
                    symbol="CLPP",
                    hpos={
                        Phenotype("HP:0000013", name=None),
                        Phenotype("HP:0000815", name=None),
                        Phenotype("HP:0000786", name=None),
                        Phenotype("HP:0000007", name=None),
                    },
                    disease_ids={
                        "OMIM:614129",
                    },
                ),
                "1000": Gene(
                    symbol="ABCD",
                    hpos={
                        Phenotype("HP:0000001", name=None),
                    },
                    disease_ids={"OMIM:614129"},
                ),
                ...
            }

        """
        gene_hpo_url = self.config["FILE"]["GENE_TO_HPO"]

        data = requests.get(gene_hpo_url).text.split("\n")
        columns = data[0].lstrip("#Format: ").split("<tab>")
        col2idx = {col: idx for idx, col in enumerate(columns)}

        gene_info = dict()
        for line in data[1:]:
            if not line:
                continue

            row = line.strip().split("\t")
            id = row[col2idx["entrez-gene-id"]]
            hpo_id = Phenotype(row[col2idx["HPO-Term-ID"]])
            disease_id = row[col2idx["disease-ID for link"]]

            if id not in gene_info:
                gene_info[id] = Gene(
                    symbol=row[col2idx["entrez-gene-symbol"]],
                    hpos={hpo_id},
                    disease_ids={disease_id},
                )
                continue

            gene_info[id].hpos.add(hpo_id)
            gene_info[id].disease_ids.add(disease_id)

        self.gene_info = gene_info

        return

    def get_gene_patient_symtom_similarity(
        self, patient_hpos: Set[Phenotype], return_symbol=False
    ) -> Dict[str, float]:
        """유전자의 표현형과 환자의 증상(=표현형)에 유사성을 계산함

        Args:
            patient_hpos (Set[Phenotype]): 환자의 증상
            return_symbol (bool): 심볼로 key을 반환할 경우

        Raises:
            NameError: gene_info가 미리 계산이 안된 경우의 에러

        Returns:
            Dict: 유전자별 - 유사성점수
        """
        if not self.gene_info:
            msg = (
                "the instnace variable ('gene_info') was not defined. "
                "Call self.set_gene_info() method."
            )
            raise NameError(msg)

        gene_similarities = dict()
        for gene_id, gene in self.gene_info.items():
            gene_similarity = self.get_semantic_similarity_one_side(gene.hpos, patient_hpos)
            gene_similarity += self.get_semantic_similarity_one_side(patient_hpos, gene.hpos)

            if return_symbol:
                gene: Gene = self.gene_info[gene_id]
                gene_similarities[gene.symbol] = gene_similarity / 2
                continue

            gene_similarities[gene_id] = gene_similarity / 2

        return gene_similarities

    def get_disease_similarty(self, patient_hpos: Set[Phenotype]) -> Dict[str, float]:
        """환자의 증상유사도 점수를 유전자 유사성의 Upweighting하여 계산

        Args:
            patient_hpos (Set[Phenotype]): 환자의 증상

        Returns:
            dict: 질환별 증상유사도
        """
        disease_similarities = dict()
        for disease_id, disease in self.disease_pheno_map.items():
            disease_similarities[disease_id] = self.get_semantic_similarity(
                patient_hpos, disease.pheno
            )

        gene_similarity = self.get_gene_patient_symtom_similarity(patient_hpos)

        gene_list = list()
        gene_similarty_list = list()
        for gene_id, similarity in gene_similarity.items():
            gene_list.append(self.gene_info[gene_id])
            gene_similarty_list.append(similarity)

        norm_gene_similarity = softmax(gene_similarty_list)
        ordered_index = np.argsort(norm_gene_similarity)[::-1]

        # Gene표현형-환자표현형의 점수가 높은 순서대로
        upweighted_disease_ids = set()
        for gene_idx in ordered_index:
            upweight = norm_gene_similarity[gene_idx] + 1
            gene: Gene = gene_list[gene_idx]
            for disease_id in gene.disease_ids:
                if disease_id in upweighted_disease_ids or disease_id not in disease_similarities:
                    continue

                disease_similarities[disease_id] = disease_similarities[disease_id] * upweight
                upweighted_disease_ids.add(disease_id)

        return disease_similarities


class Phen2DiseaseCalculator(BaseSimilarityCalculator):
    """Phen2Disease의 구현

    Note:
        https://academic.oup.com/bib/article-abstract/24/4/bbad172/7185480?redirectedFrom=fulltext

    Example:
        >>> from SemanticSimilarity.build_datasets import OntologyHandler
        >>> handler = OntologyHandler(
                OmegaConf.load(".../config.yaml"),
            )
        >>> handler.set_disease_pheno_map()

        >>> from SemanticSimilarity.calculator import Pheno2DiseaseCalculator
        >>> calculator = Pheno2DiseaseCalculator(
                OmegaConf.load(".../config.yaml")
            )

        >>> calculator.set_disease_pheno_map(handler.disease_pheno_map)
        >>> phenotypes = {Phenotype(_id) for _id in ["HP:0000093", "HP:0002907", "HP:0012211"]}
        >>> calculator.get_disease_similarity(phenotypes)
        {
            'OMIM:607832', 1.6670005975541418),
            'ORPHA:2613', 1.657498501977634),
            'OMIM:617783', 1.5869214759903247),
            'OMIM:310468', 1.5853587874216544),
            'ORPHA:84090', 1.5712670060013652),
            ...
        }

    """

    def __init__(self, config, hpo_ontology=None, logger=logging.Logger(__name__)) -> None:
        super().__init__(config, hpo_ontology, logger)

        self.set_level()
        self.set_mica_mat()

    def cal_lin_similarity(self, phenotype1: Phenotype, phenotype2: Phenotype) -> float:
        try:
            mica: Phenotype = self.get_most_informative_common_ancestor(phenotype1, phenotype2)
        except:
            return 0

        return 2 * self.ic[mica.id] / (self.ic[phenotype1.id] + self.ic[phenotype2.id])

    def get_lin_numberator_denominator(
        self, phenotypes1: Set[Phenotype], phenotypes2: Set[Phenotype]
    ) -> Tuple[float, float]:
        numerator = 0
        denominator = 0
        for phenotype1 in phenotypes1:
            p_ic = self.ic[phenotype1.id]
            denominator += p_ic

            max_sim = 0
            for phenotype2 in phenotypes2:
                similarity = self.cal_lin_similarity(phenotype1, phenotype2)
                if max_sim < similarity:
                    max_sim = similarity

            numerator += max_sim * p_ic

        return numerator, denominator

    def cal_patient_centered_similarty(
        self, p_phenotypes: Set[Phenotype], disease_phenotypes: Set[Phenotype]
    ) -> float:
        numerator, denominator = self.get_lin_numberator_denominator(
            p_phenotypes, disease_phenotypes
        )

        return numerator / denominator

    def get_semantic_similarity(
        self, p_phenotypes: Set[Phenotype], d_phenotypes: Set[Phenotype]
    ) -> float:
        p_numerator, p_denominator = self.get_lin_numberator_denominator(p_phenotypes, d_phenotypes)
        d_numerator, d_denominator = self.get_lin_numberator_denominator(d_phenotypes, p_phenotypes)
        patient_disease_similarity = (p_numerator + d_numerator) / (p_denominator + d_denominator)

        patient_similiarity = self.cal_patient_centered_similarty(p_phenotypes, d_phenotypes)

        return patient_disease_similarity + patient_similiarity
