import os
import sys
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SEMANTIC_SIM_DIR = os.path.dirname(TESTS_DIR)
sys.path.append(SEMANTIC_SIM_DIR)
from data_model import Phenotype, Gene, HPOIdFormatError, Patient, Disease
from pronto import Ontology


def test_check_valid_hpo_pattern():
    with pytest.raises(HPOIdFormatError):
        Phenotype("HP:123456")


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

    t5 = onto_hpo.create_term("HP:0040068")
    t5.name = "Abnormality of the limb bone"
    t5.superclasses().add(t2)
    t5.superclasses().add(t4)

    return onto_hpo


@pytest.mark.parametrize(
    "id, expected",
    [
        pytest.param("HP:0000001", True),
        pytest.param("HP:0000002", False),
        pytest.param("HP:0040068", True),
    ],
)
def test_check_id_in_ontology(id, expected, ontology):
    assert expected == Phenotype(id=id).check_id_in_ontology(ontology)


@pytest.mark.parametrize(
    "phenotype1, phenotype2, expected",
    [
        pytest.param(Phenotype("HP:0000001"), Phenotype("HP:0000001"), True),
        pytest.param(Phenotype("HP:0000001"), Phenotype("HP:0000002"), False),
    ],
)
def test_eq_phenotype(phenotype1, phenotype2, expected):
    assert expected == (phenotype1 == phenotype2)


@pytest.mark.parametrize(
    "gene1, gene2, expected",
    [
        pytest.param(
            Gene(
                symbol="ACL",
                hpos={Phenotype("HP:0000001"), Phenotype("HP:0000001")},
                disease_ids={"OMIM:00001"},
            ),
            Gene(
                symbol="ACL",
                hpos={Phenotype("HP:0000001"), Phenotype("HP:0000001")},
                disease_ids={"OMIM:00001"},
            ),
            True,
            id="TEST1: All identical attributes",
        ),
        pytest.param(
            Gene(
                symbol="ACL",
                hpos={Phenotype("HP:0000001"), Phenotype("HP:0000001")},
                disease_ids={"OMIM:00001"},
            ),
            Gene(
                symbol="AC3L",
                hpos={Phenotype("HP:0000001"), Phenotype("HP:0000001")},
                disease_ids={"OMIM:00001"},
            ),
            False,
            id="TEST2: different symbol",
        ),
    ],
)
def test_eq_gene(gene1, gene2, expected):
    assert expected == (gene1 == gene2)


@pytest.mark.parametrize(
    "patients",
    [
        pytest.param(
            Patient(
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
        )
    ],
)
def test_patient_to_json(patients):
    res = patients.to_json()
    assert patients.id == res["id"]
    for query_dis, res_dis in zip(patients.diseases.keys(), res["diseases"]):
        assert query_dis == res_dis["disease_id"]
