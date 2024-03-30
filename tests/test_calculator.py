import os
import sys
import math
import pytest
from unittest.mock import patch, Mock

import yaml
import pandas as pd
import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SEMANTIC_SIM_DIR = os.path.dirname(TESTS_DIR)
sys.path.append(SEMANTIC_SIM_DIR)
from data_model import Phenotype, Gene, Disease
from calculator import (
    cal_symtpom_similarity_from_lambda,
    BaseSimilarityCalculator,
    ResnikSimilarityCalculator,
    NodeLevelSimilarityCalculator,
    FrequentBasedSimilarityCaculator,
    GeneDiseaseSimilarityCalculator,
)
from pronto import Ontology


@patch("requests.post")
def test_cal_symtpom_similarity_from_lambda(mock_post):
    # Mock response from the AWS Lambda service
    mock_response = Mock()
    mock_post.return_value = mock_response
    mock_response.json.return_value = {"body": ["PMID:12345,0.75,_", "67890,0.60,_"]}

    hpos = ["HP:0000198", "HP:0000564"]
    expected = {
        "3B:12345": 0.75,
        "OMIM:67890": 0.60,
    }

    assert expected == cal_symtpom_similarity_from_lambda(hpos, timeout=120)


@pytest.fixture(scope="module")
def ontology():
    onto_hpo = Ontology()

    root = onto_hpo.create_term("HP:0000001")
    root.name = "All"

    t1 = onto_hpo.create_term("HP:0000118")
    t1.name = "Phenotypic abnormality"
    t1.superclasses().add(root)

    t2 = onto_hpo.create_term("HP:0040064")
    t2.name = "Abnormality of limbs"
    t2.superclasses().add(t1)

    t3 = onto_hpo.create_term("HP:0033127")
    t3.name = "Abnormality of musculoskeletal system"
    t3.superclasses().add(t1)

    t4 = onto_hpo.create_term("HP:0000924")
    t4.name = "Abnormality of the skeletal system"
    t4.superclasses().add(t3)

    t5 = onto_hpo.create_term("HP:0011842")
    t5.name = "Abnormality of the skeletal morphology"
    t5.superclasses().add(t4)

    t6 = onto_hpo.create_term("HP:0040068")
    t6.name = "Abnormality of the limb bone"
    t6.superclasses().add(t2)
    t6.superclasses().add(t5)

    return onto_hpo


@pytest.fixture(scope="module")
def config():
    with open(os.path.join(SEMANTIC_SIM_DIR, "config.yaml")) as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)


@pytest.fixture(scope="module")
def base_similarity_calculator(config, ontology):
    with patch("pronto.Ontology"), patch("io.BytesIO"), patch("requests.get"):
        calculator = BaseSimilarityCalculator(config, hpo_ontology=ontology)

    return calculator


def test_build_subclass_map(ontology, base_similarity_calculator):
    subclass_map = base_similarity_calculator.build_subclass_map()

    assert len(subclass_map) == len(ontology.terms())

    for term in ontology.terms():
        subterm_ids_gt = ontology[term.id].subclasses(with_self=True).to_set().ids
        assert len(subclass_map[term.id]) == len(subterm_ids_gt)


def test_set_level(base_similarity_calculator):
    base_similarity_calculator.set_level()

    expected = {
        "HP:0000001": 1,
        "HP:0000118": 2,
        "HP:0040064": 3,
        "HP:0033127": 3,
        "HP:0000924": 4,
        "HP:0011842": 5,
        "HP:0040068": 6,
    }
    assert expected == base_similarity_calculator.phenotype2level


def test_set_mica(base_similarity_calculator):
    base_similarity_calculator.subclass_map = base_similarity_calculator.build_subclass_map()
    base_similarity_calculator.phenotype2level = {
        "HP:0000001": 1,
        "HP:0000118": 2,
        "HP:0040064": 3,
        "HP:0033127": 3,
        "HP:0000924": 4,
        "HP:0011842": 5,
        "HP:0040068": 6,
    }
    base_similarity_calculator.set_mica_mat()
    expected = np.array(
        [
            [
                "HP:0000001",  # 기준: HP:0000001
                "HP:0000001",
                "HP:0000001",
                "HP:0000001",
                "HP:0000001",
                "HP:0000001",
                "HP:0000001",
            ],
            [
                "HP:0000001",  # 기준: HP:0000118
                "HP:0000118",
                "HP:0000118",
                "HP:0000118",
                "HP:0000118",
                "HP:0000118",
                "HP:0000118",
            ],
            [
                "HP:0000001",  # 기준: HP:0040064
                "HP:0000118",
                "HP:0040064",
                "HP:0000118",
                "HP:0000118",
                "HP:0000118",
                "HP:0040064",
            ],
            [
                "HP:0000001",  # 기준: HP:0033127
                "HP:0000118",
                "HP:0000118",
                "HP:0033127",
                "HP:0033127",
                "HP:0033127",
                "HP:0033127",
            ],
            [
                "HP:0000001",  # 기준: HP:0000924
                "HP:0000118",
                "HP:0000118",
                "HP:0033127",
                "HP:0000924",
                "HP:0000924",
                "HP:0000924",
            ],
            [
                "HP:0000001",  # 기준: HP:0011842
                "HP:0000118",
                "HP:0000118",
                "HP:0033127",
                "HP:0000924",
                "HP:0011842",
                "HP:0011842",
            ],
            [
                "HP:0000001",  # 기준: HP:0040068
                "HP:0000118",
                "HP:0040064",
                "HP:0033127",
                "HP:0000924",
                "HP:0011842",
                "HP:0040068",
            ],
        ]
    )

    np.testing.assert_array_equal(expected, base_similarity_calculator.mica_mat)


@pytest.mark.parametrize(
    "phenotype1, phenotype2, expected",
    [
        pytest.param(
            Phenotype("HP:0000001"),
            Phenotype("HP:0000001"),
            Phenotype("HP:0000001"),
            id="TEST1: SAME ROOT NODE",
        ),
        pytest.param(
            Phenotype("HP:0000001"),
            Phenotype("HP:0000118"),
            Phenotype("HP:0000001"),
            id="TEST2: MICA as root",
        ),
        pytest.param(
            Phenotype("HP:0040064"),
            Phenotype("HP:0000924"),
            Phenotype("HP:0000118"),
            id="TEST3: level 2 NODE",
        ),
    ],
)
def test_get_most_informative_common_ancestor(
    phenotype1, phenotype2, expected, base_similarity_calculator
):
    base_similarity_calculator.set_mica_mat()
    assert (
        expected.id
        == base_similarity_calculator.get_most_informative_common_ancestor(
            phenotype1, phenotype2
        ).id
    )


@pytest.mark.parametrize(
    "phenotype, expected",
    [
        pytest.param(Phenotype("HP:0000001"), 0, id="TEST1: ROOT NODE"),
        pytest.param(Phenotype("HP:0040068"), 1.945, id="TEST2: LEAF NODE"),
        pytest.param(Phenotype("HP:0000118"), 0.154, id="TEST3"),
        pytest.param(Phenotype("HP:0040064"), 1.252, id="TEST4 "),
        pytest.param(Phenotype("HP:0000924"), 0.847, id="TEST5 "),
    ],
)
def test_get_information_content(phenotype, expected, base_similarity_calculator):
    result = base_similarity_calculator.get_information_content(phenotype)
    assert expected == pytest.approx(result, 0.01)


@pytest.fixture(scope="module")
def resnik_similarity_calculator(config, ontology):
    with patch("pronto.Ontology"), patch("io.BytesIO"), patch("requests.get"):
        calculator = ResnikSimilarityCalculator(config, hpo_ontology=ontology)

    return calculator


@pytest.mark.parametrize(
    "phenotypes1, phenotypes2, expected",
    [
        pytest.param(
            {Phenotype("HP:0000001")},
            {Phenotype("HP:0000001")},
            0,
            id="TEST1: identical graph",
        ),
        pytest.param(
            {Phenotype("HP:0000001"), Phenotype("HP:0000118")},
            {Phenotype("HP:0000001")},
            0,
            id="TEST2: All MICA are root",
        ),
        pytest.param(
            {Phenotype("HP:0040064"), Phenotype("HP:0000924")},
            {Phenotype("HP:0040064"), Phenotype("HP:0000924")},
            1.0495,
            id="TEST3: identical graph",
        ),
        pytest.param(
            {Phenotype("HP:0040064")},
            {
                Phenotype("HP:0000118"),
                Phenotype("HP:0000924"),
            },
            0.1541,
            id="TEST4: identical graph",
        ),
        pytest.param(
            {
                Phenotype("HP:0000118"),
                Phenotype("HP:0000924"),
            },
            {Phenotype("HP:0040064")},
            0.1541,
            id="TEST5: Opposite side of TEST4",
        ),
        pytest.param(
            {
                Phenotype("HP:0000118"),
                Phenotype("HP:0000924"),
                Phenotype("HP:0040064"),
            },
            {Phenotype("HP:0040064")},
            0.5203,
            id="TEST6: Opposite side of TEST4",
        ),
        pytest.param(
            {Phenotype("HP:0040064")},
            {
                Phenotype("HP:0000118"),
                Phenotype("HP:0000924"),
                Phenotype("HP:0040064"),
            },
            1.2527,
            id="TEST7: Opposite side of TEST6",
        ),
    ],
)
def test_get_semantic_similarity_one_side_resnik(
    phenotypes1, phenotypes2, expected, resnik_similarity_calculator
):
    resnik_similarity_calculator.subclass_map = resnik_similarity_calculator.build_subclass_map()
    resnik_similarity_calculator.set_level()
    resnik_similarity_calculator.set_mica_mat()
    result = resnik_similarity_calculator.get_semantic_similarity_one_side(phenotypes1, phenotypes2)
    assert expected == pytest.approx(result, 0.01)


@pytest.fixture(scope="module")
def node_level_similarity_calculator(config, ontology):
    with patch("pronto.Ontology"), patch("io.BytesIO"), patch("requests.get"):
        calculator = NodeLevelSimilarityCalculator(config, hpo_ontology=ontology)

    return calculator


@pytest.mark.parametrize(
    "phenotypes1, phenotypes2, expected",
    [
        pytest.param(
            {Phenotype("HP:0000001")},
            {Phenotype("HP:0000001")},
            1,
            id="TEST1: identical graph",
        ),
        pytest.param(
            {Phenotype("HP:0000001"), Phenotype("HP:0000118")},
            {Phenotype("HP:0000001")},
            1,
            id="TEST2: All MICA are root",
        ),
        pytest.param(
            {
                Phenotype("HP:0040064"),
                Phenotype("HP:0000924"),
            },  # MAX[MICA(0040064,0040064) MICA(0040064, 0000924)]]=3 + MAX[MICA(0000924,0040064) MICA(0000924, 0000924)]]=4
            {Phenotype("HP:0040064"), Phenotype("HP:0000924")},
            3.5,
            id="TEST3: identical graph",
        ),
        pytest.param(
            {Phenotype("HP:0040064")},  # MAX[MICA(0040064,0000118) MICA(0040064, 0000924)]]=2
            {Phenotype("HP:0000118"), Phenotype("HP:0000924")},
            2,
            id="TEST4: 1 Parent + 1 other relationship",
        ),
        pytest.param(
            {
                Phenotype("HP:0000118"),
                Phenotype("HP:0000924"),
            },  # level(MICA(0000118, 0040064))=2, level(MICA(0000924,0040064))=2
            {Phenotype("HP:0040064")},
            2,
            id="TEST5: Opposite side of TEST4",
        ),
        pytest.param(
            {
                Phenotype("HP:0000118"),
                Phenotype("HP:0000924"),
                Phenotype("HP:0040064"),
            },  # MICA:(0000118, 0040064)=2, MICA(0000924, 0040064)=2, MICA(0040064, 0040064)=3
            {Phenotype("HP:0040064")},
            2.3333,
            id="TEST6: Additional Term of test 5",
        ),
        pytest.param(
            {Phenotype("HP:0040064")},
            {
                Phenotype("HP:0000118"),
                Phenotype("HP:0000924"),
                Phenotype("HP:0040064"),
            },  # exact match
            3,
            id="TEST7: Opposite side of TEST6",
        ),
    ],
)
def test_get_semantic_similarity_one_side_node_level(
    phenotypes1, phenotypes2, expected, node_level_similarity_calculator
):
    node_level_similarity_calculator.subclass_map = (
        node_level_similarity_calculator.build_subclass_map()
    )
    node_level_similarity_calculator.set_level()
    node_level_similarity_calculator.set_mica_mat()
    result = node_level_similarity_calculator.get_semantic_similarity_one_side(
        phenotypes1, phenotypes2
    )
    assert expected == pytest.approx(result, 0.01)


@pytest.fixture(scope="module")
def frequency_based_calculator(config, ontology):
    with patch("pronto.Ontology"), patch("io.BytesIO"), patch("requests.get"):
        calculator = FrequentBasedSimilarityCaculator(config, hpo_ontology=ontology)

    return calculator


def test_set_disease_hpo(config, frequency_based_calculator):
    with patch(
        "pandas.read_csv",
        return_value=pd.DataFrame(
            data=[
                [
                    "OMIM:619340",
                    "HP:0011090",
                    "Developmental and epileptic encephalopathy 96",
                    "3/8",
                ],
                [
                    "OMIM:619340",
                    "HP:0011091",
                    "Developmental and epileptic encephalopathy 96",
                    math.nan,
                ],
                [
                    "OMIM:619340",
                    "HP:0011092",
                    "Developmental and epileptic encephalopathy 96",
                    "50%",
                ],
                [
                    "OMIM:999999",
                    "HP:0011092",
                    "TEST disease name",
                    "HP:0040283",
                ],
            ],
            columns=config["COLS"]["DISEASE_HPO"],
        ),
    ):
        frequency_based_calculator.set_disease_hpo()

    expected = {
        "OMIM:619340": {
            "HP:0011090": 0,
            "HP:0011091": 0,
            "HP:0011092": 0.5,
        },
        "OMIM:999999": {"HP:0011092": 0},
    }
    assert expected == frequency_based_calculator.disease_hpos


@pytest.mark.parametrize(
    "patient_hpo, disease_id, disease_hpos, expected",
    [
        pytest.param(
            {Phenotype("HP:0040064"), Phenotype("HP:0000924")},
            "OMIM:111111",
            {
                "OMIM:111111": {
                    "HP:0040064": 0.5,
                }
            },
            1.7034,
            id="TEST1",
        ),
        pytest.param(
            {Phenotype("HP:0040064"), Phenotype("HP:0000924")},
            "OMIM:111111",
            {
                "OMIM:111111": {
                    "HP:0040064": 0.5,  # IC(MICA(0040064, 0040064))=1.098, IC(MICA(0040064, 0040068))]=1.098 // IC(MICA(0000924, 0040064))=0.182, IC(MICA(0000924, 0040068))]=1.098
                    "HP:0040068": 0.0,
                }
            },
            1.7034,  # ((1.098) * 1.5 + (1.098*1.0))/2
            id="TEST2: (1.647 + 0.273) / 2",
        ),
        pytest.param(
            {Phenotype("HP:0040064"), Phenotype("HP:0000924")},
            "OMIM:222222",
            {
                "OMIM:222222": {
                    "HP:0040064": 0.05,  # IC(MICA(0040064, 0040064))=1.098, IC(MICA(0040064, 0040068))]=1.098m IC(MICA(0000924, 0000924))=1.098 // IC(MICA(0000924, 0040064))=0.182, IC(MICA(0000924, 0040068))]=1.098, IC(MICA(0000924, 0000924))=1.098
                    "HP:0040068": 0.05,
                    "HP:0000924": 0.05,
                }
            },
            1.3833,  # ((1.098) * 1.05 + (1.098*1.05))/2
            id="TEST3: underweighted frequency",
        ),
    ],
)
def test_get_frequency_semantic_similarity_one_side(
    patient_hpo, disease_id, disease_hpos, expected, frequency_based_calculator
):
    frequency_based_calculator.subclass_map = frequency_based_calculator.build_subclass_map()
    frequency_based_calculator.set_level()
    frequency_based_calculator.set_mica_mat()
    frequency_based_calculator.disease_hpos = disease_hpos
    result = frequency_based_calculator.get_frequency_semantic_similarity_one_side(
        patient_hpo, disease_id
    )
    assert expected == pytest.approx(result, 0.05)


@pytest.mark.parametrize(
    "phenotypes1, phenotypes2, expected",
    [
        pytest.param(
            {Phenotype("HP:0000001")},
            {Phenotype("HP:0000001")},
            0,
            id="TEST1: identical graph",
        ),
        pytest.param(
            {Phenotype("HP:0000001"), Phenotype("HP:0000118")},
            {Phenotype("HP:0000001")},
            0,
            id="TEST2: All MICA are root",
        ),
        pytest.param(
            {Phenotype("HP:0040064"), Phenotype("HP:0000924")},
            {Phenotype("HP:0040064"), Phenotype("HP:0000924")},
            1.0495,
            id="TEST3: identical graph",
        ),
        pytest.param(
            {Phenotype("HP:0040064")},
            {
                Phenotype("HP:0000118"),
                Phenotype("HP:0000924"),
            },
            0.1541,
            id="TEST4: identical graph",
        ),
        pytest.param(
            {
                Phenotype("HP:0000118"),
                Phenotype("HP:0000924"),
            },
            {Phenotype("HP:0040064")},
            0.1541,
            id="TEST5: Opposite side of TEST4",
        ),
        pytest.param(
            {
                Phenotype("HP:0000118"),
                Phenotype("HP:0000924"),
                Phenotype("HP:0040064"),
            },
            {Phenotype("HP:0040064")},
            0.5203,
            id="TEST6: Opposite side of TEST4",
        ),
        pytest.param(
            {Phenotype("HP:0040064")},
            {
                Phenotype("HP:0000118"),
                Phenotype("HP:0000924"),
                Phenotype("HP:0040064"),
            },
            1.2527,
            id="TEST7: Opposite side of TEST6",
        ),
    ],
)
def test_freq_get_semantic_similarity_one_side(
    phenotypes1, phenotypes2, expected, frequency_based_calculator
):
    frequency_based_calculator.subclass_map = frequency_based_calculator.build_subclass_map()
    frequency_based_calculator.set_level()
    frequency_based_calculator.set_mica_mat()

    result = frequency_based_calculator.get_semantic_similarity_one_side(phenotypes1, phenotypes2)
    assert expected == pytest.approx(result, 0.01)


@pytest.fixture()
def gene_disease_similarity_calculator(config, ontology):
    with patch("pronto.Ontology"), patch("io.BytesIO"), patch("requests.get"):
        calculator = GeneDiseaseSimilarityCalculator(config, hpo_ontology=ontology)
    return calculator


@patch("requests.get")
def test_set_gene_info(
    mock_get,
    gene_disease_similarity_calculator,
):
    mock_get.return_value.text = (
        "entrez-gene-id<tab>entrez-gene-symbol<tab>HPO-Term-ID<tab>HPO-Term-Name<tab>Frequency-Raw<tab>Frequency-HPO<tab>Additional Info from G-D source<tab>G-D source<tab>disease-ID for link\n"
        "8192\tCLPP\tHP:0000013\tHypoplasia of the uterus\t-\t\t-\tmim2gene\tOMIM:614129\n"
        "8192\tCLPP\tHP:0000815\tHypergonadotropic hypogonadism\t\t-\t-\tmim2gene\tOMIM:614129\n"
        "8192\tCLPP\tHP:0000786\tPrimary amenorrhea\t-\t\t-\tmim2gene\tOMIM:614129\n"
        "8192\tCLPP\tHP:0000007\tAutosomal recessive inheritance\t-\t-\t\tmim2gene\tOMIM:614129\n"
        "1000\tABCD\tHP:0000001\tDisease name\t-\t-\t\tmim2gene\tOMIM:614129"
    )
    gene_disease_similarity_calculator.subclass_map = (
        gene_disease_similarity_calculator.build_subclass_map()
    )
    gene_disease_similarity_calculator.set_level()
    gene_disease_similarity_calculator.set_mica_mat()
    gene_disease_similarity_calculator.set_gene_info()

    expected = {
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
    }

    result = gene_disease_similarity_calculator.gene_info

    for gene_id, gene in expected.items():
        assert gene == result[gene_id]


@pytest.mark.parametrize(
    "patient_hpos, expected",
    [
        pytest.param(
            {Phenotype("HP:0000001")},
            {
                "8192": 0,
                "1000": 0,
                "2000": 0,
            },
        ),
        pytest.param(
            {Phenotype("HP:0000001"), Phenotype("HP:0040064")},
            {
                "8192": 0.8239,
                "1000": 0,
                "2000": 0.8239,
            },
        ),
    ],
)
def test_get_gene_patient_symtom_similarity(
    patient_hpos, expected, gene_disease_similarity_calculator
):
    gene_disease_similarity_calculator.subclass_map = (
        gene_disease_similarity_calculator.build_subclass_map()
    )
    gene_disease_similarity_calculator.set_level()
    gene_disease_similarity_calculator.set_mica_mat()
    gene_disease_similarity_calculator.gene_info = {
        "8192": Gene(
            symbol="CLPP",
            hpos={
                Phenotype("HP:0040068"),
            },
            disease_ids={
                "OMIM:000001",
            },
        ),
        "1000": Gene(
            symbol="ABCD",
            hpos={
                Phenotype("HP:0000001"),
            },
            disease_ids={"OMIM:000002"},
        ),
        "2000": Gene(
            symbol="ABCD",
            hpos={
                Phenotype("HP:0040064"),
            },
            disease_ids={"OMIM:000003"},
        ),
    }

    result = gene_disease_similarity_calculator.get_gene_patient_symtom_similarity(patient_hpos)

    for gene_id, similarity in result.items():
        expected[gene_id] == pytest.approx(similarity, 0.05)


def test_get_disease_similarty(gene_disease_similarity_calculator):
    gene_disease_similarity_calculator.gene_info = {
        "1000": Gene(
            symbol="CLPP",
            hpos={
                Phenotype("HP:0040068"),
            },
            disease_ids={
                "OMIM:000001",
            },
        ),
        "2000": Gene(
            symbol="ABCD",
            hpos={
                Phenotype("HP:0000001"),
            },
            disease_ids={"OMIM:000002"},
        ),
        "3000": Gene(
            symbol="QWER",
            hpos={
                Phenotype("HP:0040064"),
            },
            disease_ids={"OMIM:000003"},
        ),
        "4000": Gene(
            symbol="ASDF",
            hpos={
                Phenotype("HP:0040064"),
            },
            disease_ids={"OMIM:000001"},  # 다른유전자 같은 질환 사례
        ),
    }

    gene_disease_similarity_calculator.disease_pheno_map = {
        "OMIM:000001": Disease(
            disease_name="test_diseas_name",
            disease_id="OMIM:000001",
            pheno={Phenotype(id="HP:0000001")},
        ),
        "OMIM:000002": Disease(
            disease_name="test_diseas_name",
            disease_id="OMIM:000001",
            pheno={Phenotype(id="HP:0000001")},
        ),
        "OMIM:000003": Disease(
            disease_name="test_diseas_name",
            disease_id="OMIM:000001",
            pheno={Phenotype(id="HP:0000001")},
        ),
    }
    gene_disease_similarity_calculator.get_semantic_similarity = Mock(
        side_effect=[
            3.2,
            1.2,
            1.9,
        ]
    )
    gene_disease_similarity_calculator.get_gene_patient_symtom_similarity = Mock(
        return_value={"1000": 1.0, "2000": 3.5, "3000": 5.0, "4000": 0.5}
    )  # [0.01462262, 0.17814004, 0.79836827, 0.00886907] == softmax([1.0, 3.5, 5.0, 0.5])

    patient_hpos = {Phenotype("HP:0000001")}

    result = gene_disease_similarity_calculator.get_disease_similarty(patient_hpos)
    expected = {
        "OMIM:000001": 3.2 * 1.01462262,
        "OMIM:000002": 1.2 * 1.17814004,
        "OMIM:000003": 1.9 * 1.79836827,
    }
    assert expected == pytest.approx(result, 0.005)
