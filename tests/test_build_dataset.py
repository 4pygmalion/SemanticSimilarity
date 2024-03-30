import os
import sys
import pytest
import pandas as pd
import json

from omegaconf import OmegaConf
from pronto import Ontology
from unittest.mock import MagicMock, patch

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SEMANTIC_DIR = os.path.dirname(TEST_DIR)
ROOT_DIR = os.path.dirname(SEMANTIC_DIR)
sys.path.append(ROOT_DIR)

from SemanticSimilarity.data_model import Patient, Phenotype, Disease
from SemanticSimilarity.build_datasets import (
    OntologyHandler,
    PhenotypeIdNotInOntologyException,
    TypeMissmatchException,
)


@pytest.fixture(scope="module")
def disease_pheno_df():
    df = pd.DataFrame([], columns=["#DatabaseID", "DiseaseName", "HPO_ID"])
    df["#DatabaseID"] = [
        "OMIM:619340",
        "OMIM:619340",
        "OMIM:619341",
        "OMIM:619341",
        "ORPHA:231160",
        "ORPHA:231160",
        "ORPHA:231160",
        "ORPHA:231160",
        "ORPHA:231160",
    ]
    df["DiseaseName"] = [
        "Developmental and epileptic encephalopathy 96",
        "Developmental and epileptic encephalopathy 96",
        "Developmental and epileptic encephalopathy 100",
        "Developmental and epileptic encephalopathy 100",
        "Familial cerebral saccular aneurysm",
        "Familial cerebral saccular aneurysm",
        "Familial cerebral saccular aneurysm",
        "Familial cerebral saccular aneurysm",
        "Familial cerebral saccular aneurysm",
    ]
    df["HPO_ID"] = [
        "HP:0011097",
        "HP:0002187",
        "HP:0001518",
        "HP:0032792",
        "HP:0002170",
        "HP:0002616",
        "HP:0002647",
        "HP:0012246",
        "HP:0040197",
    ]
    return df


@pytest.fixture(scope="module")
def ontology():
    onto_hpo = Ontology()

    root = onto_hpo.create_term("HP:0011097")
    root.name = "Epileptic spasm"

    t1 = onto_hpo.create_term("HP:0002187")
    t1.name = "Intellectual disability, profound"
    t1.superclasses().add(root)

    t2 = onto_hpo.create_term("HP:0001518")
    t2.name = "Small for gestational age"
    t2.superclasses().add(t1)

    t3 = onto_hpo.create_term("HP:0032792")
    t3.name = "Tonic seizure"
    t3.superclasses().add(t2)

    t4 = onto_hpo.create_term("HP:0002170")
    t4.name = "Intracranial hemorrhage"
    t4.superclasses().add(t3)

    t5 = onto_hpo.create_term("HP:0002616")
    t5.name = "Aortic root aneurysm"
    t5.superclasses().add(t4)

    t6 = onto_hpo.create_term("HP:0002647")
    t6.name = "Aortic dissection"
    t6.superclasses().add(t5)

    t7 = onto_hpo.create_term("HP:0012246")
    t7.name = "Oculomotor nerve palsy"
    t7.superclasses().add(t6)

    t8 = onto_hpo.create_term("HP:0040197")
    t8.name = "Encephalomalacia"
    t8.superclasses().add(t7)

    return onto_hpo


@pytest.fixture(scope="module")
def config():
    return OmegaConf.load(os.path.join(SEMANTIC_DIR, "config.yaml"))


@pytest.fixture(scope="module")
@patch("SemanticSimilarity.build_datasets.OntologyHandler.set_ontology")
@patch("SemanticSimilarity.build_datasets.OntologyHandler.set_disease_pheno_map")
def ontology_handler(mock_set_disease_pheno_map, mock_set_ontology, config, ontology):
    ontology_handler = OntologyHandler(config)
    ontology_handler.ontology = ontology
    return ontology_handler


@patch("os.path.exists", return_value=True)
@patch("requests.get")
@patch("SemanticSimilarity.build_datasets.Ontology")
def test_set_ontology_file(
    mock_ontology, mock_request_get, mock_exists, ontology_handler, ontology
):
    from build_datasets import ROOT_DIR

    ontology_handler.set_ontology()
    mock_ontology.assert_called_with(os.path.join(ROOT_DIR, "data/hp.obo"))

    mock_request_get.assert_not_called()

    ontology_handler.ontology = ontology


@patch("os.path.exists", return_value=False)
@patch("requests.get")
@patch("SemanticSimilarity.build_datasets.Ontology")
@patch("io.BytesIO")
def test_set_ontology_request(
    mock_bytesio,
    mock_ontology,
    mock_request_get,
    mock_exists,
    ontology_handler,
    ontology,
):
    from build_datasets import ROOT_DIR

    ontology_handler.set_ontology()
    assert mock_ontology.call_args_list[0] != os.path.join(ROOT_DIR, "data/hp.obo")

    mock_request_get.assert_called_once()
    ontology_handler.ontology = ontology


@pytest.fixture(scope="function")
def phenotypes():
    phenotypes = {
        "HP:0011097": Phenotype(id="HP:0011097", name="Epileptic spasm"),
        "HP:0002187": Phenotype(id="HP:0002187", name="Intellectual disability, profound"),
        "HP:0001518": Phenotype(id="HP:0001518", name="Small for gestational age"),
        "HP:0032792": Phenotype(id="HP:0032792", name="Tonic seizure"),
        "HP:0002170": Phenotype(id="HP:0002170", name="Intracranial hemorrhage"),
        "HP:0002647": Phenotype(id="HP:0002647", name="Aortic dissection"),
        "HP:0002616": Phenotype(id="HP:0002616", name="Aortic root aneurysm"),
        "HP:0012246": Phenotype(id="HP:0012246", name="Oculomotor nerve palsy"),
        "HP:0040197": Phenotype(id="HP:0040197", name="Encephalomalacia"),
    }

    return phenotypes


@pytest.fixture(scope="function")
def diseases():
    diseases = {
        "OMIM:619340": Disease(
            disease_name="Developmental and epileptic encephalopathy 96",
            disease_id="OMIM:619340",
            pheno={
                Phenotype(name="Epileptic spasm", id="HP:0011097"),
                Phenotype(
                    name="Intellectual disability, profound",
                    id="HP:0002187",
                ),
            },
        ),
        "OMIM:619341": Disease(
            disease_name="Developmental and epileptic encephalopathy 100",
            disease_id="OMIM:619341",
            pheno={
                Phenotype(name="Small for gestational age", id="HP:0001518"),
                Phenotype(name="Tonic seizure", id="HP:0032792"),
            },
        ),
        "ORPHA:231160": Disease(
            disease_name="Familial cerebral saccular aneurysm",
            disease_id="ORPHA:231160",
            pheno={
                Phenotype(name="Encephalomalacia", id="HP:0040197"),
                Phenotype(name="Aortic dissection", id="HP:0002647"),
                Phenotype(name="Intracranial hemorrhage", id="HP:0002170"),
                Phenotype(name="Aortic root aneurysm", id="HP:0002616"),
                Phenotype(name="Oculomotor nerve palsy", id="HP:0012246"),
            },
        ),
    }

    return diseases


@pytest.mark.parametrize(
    "query, expected",
    [
        pytest.param(
            Phenotype(id="HP:0011097", name="Epileptic spasm"),
            True,
            id="success1",
        ),
        pytest.param(
            Phenotype(id="HP:1233455", name="Should return False"),
            False,
            id="success2",
        ),
    ],
)
def test_is_member_of_phenotypes(phenotypes, query, expected):
    another = set(phenotypes.keys())
    res = query.is_member_of(another)
    assert res == expected


def test_get_n_phenos_randomly(diseases):
    max_n = 10
    disease_ids = list(diseases.keys())

    for did in disease_ids:
        res = diseases[did].get_n_phenos_randomly(max_n=max_n)
        assert len(res) == min(max_n, len(res))
        assert type(res) == set
        assert res.issubset(diseases[did].pheno)


def test_get_phenotype_from_hpoid_success(ontology_handler, phenotypes):
    ontology_handler.ontology
    for hpoid, phenotype in phenotypes.items():
        result = ontology_handler.get_phenotype_from_hpoid(hpoid=hpoid)
        assert phenotype == result


@pytest.mark.parametrize(
    "query",
    [
        pytest.param({"1233445"}, id="fail1"),
        pytest.param({"asdfasdf"}, id="fail2"),
    ],
)
def test_get_phenotype_from_hpoid_fail(ontology_handler, query):
    with pytest.raises(PhenotypeIdNotInOntologyException):
        ontology_handler.get_phenotype_from_hpoid(hpoid=query)


def test_get_super_pheno_succeess(ontology_handler):
    queries = [
        {ontology_handler.ontology["HP:0011097"]},
        {ontology_handler.ontology["HP:0032792"]},
    ]
    expected = [
        set(),
        set(
            [
                "HP:0011097",
                "HP:0002187",
                "HP:0001518",
            ]
        ),
    ]
    for query, exp in zip(queries, expected):
        assert exp == ontology_handler.get_super_pheno_from(hpo_termset=query)


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            "HP:0011097",
            id="fail1",
        ),
    ],
)
def test_get_super_pheno_fail(ontology_handler, query):
    with pytest.raises(TypeMissmatchException):
        ontology_handler.get_super_pheno_from(hpo_termset=query)


def test_build_disease_pheno_map(disease_pheno_df, ontology_handler, phenotypes, diseases):
    def hpoid_side_effect(args):
        return phenotypes[args]

    ontology_handler.get_phenotype_from_hpoid = MagicMock()
    ontology_handler.get_phenotype_from_hpoid.side_effect = hpoid_side_effect

    disease_pheno_map = ontology_handler._build_disease_pheno_map(disease_pheno_df=disease_pheno_df)
    assert diseases == disease_pheno_map


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(["OMIM:619340", "OMIM:619341"], id="success1"),
        pytest.param([], id="success2"),
    ],
)
def test_build_optimal_patients_map(
    diseases,
    ontology_handler,
    query,
):
    max_pheno = 10
    ontology_handler.disease_pheno_map = diseases
    optiaml_patients_map = ontology_handler.build_optimal_patients_map(
        disease_ids=query, max_pheno=max_pheno
    )

    for disease_id in optiaml_patients_map.keys():
        assert (
            0
            < len(optiaml_patients_map[disease_id].pheno)
            <= len(ontology_handler.disease_pheno_map[disease_id].pheno)
        )


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(["OMIM:619340", "OMIM:619341"], id="success1"),
        pytest.param([], id="success2"),
    ],
)
def test_build_noisy_patients_map(diseases, ontology_handler, phenotypes, query):
    noise_ratios = [0.1, 0.2, 0.5, 0.7, 0.9]

    def hpoid_side_effect(args):
        return phenotypes[args]

    def get_super_pheno_from_side_effect(hpo_termset, with_self):
        return set(["HP:0011097", "HP:0002187", "HP:0001518"])

    ontology_handler.build_optimal_patients_map = MagicMock()
    ontology_handler.build_optimal_patients_map.return_value = diseases

    ontology_handler.all_phenos = set(phenotypes.keys())
    ontology_handler.get_phenotype_from_hpoid = MagicMock()
    ontology_handler.get_phenotype_from_hpoid.side_effect = hpoid_side_effect

    ontology_handler.get_super_pheno_from = MagicMock()
    ontology_handler.get_super_pheno_from.side_effect = get_super_pheno_from_side_effect

    for noise_ratio in noise_ratios:
        noisy_patients_map = ontology_handler.build_noisy_patients_map(
            disease_ids=query,
            noise_ratio=noise_ratio,
        )

        for disease_id in noisy_patients_map.keys():
            assert len(noisy_patients_map[disease_id].pheno) != 0


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(["OMIM:619340", "OMIM:619341"], id="success1"),
        pytest.param([], id="success2"),
    ],
)
def test_build_ambiguous_patient_map(ontology_handler, diseases, phenotypes, query):
    ambiguous_ratios = [0.1, 0.2, 0.5, 0.7, 0.9]

    def hpoid_side_effect(args):
        return phenotypes[args]

    def get_super_pheno_from_side_effect(hpo_termset):
        return set(["HP:0011097", "HP:0002187", "HP:0001518"])

    ontology_handler.build_optimal_patients_map = MagicMock()
    ontology_handler.build_optimal_patients_map.return_value = diseases

    ontology_handler.get_phenotype_from_hpoid = MagicMock()
    ontology_handler.get_phenotype_from_hpoid.side_effect = hpoid_side_effect

    ontology_handler.get_super_pheno_from = MagicMock()
    ontology_handler.get_super_pheno_from.side_effect = get_super_pheno_from_side_effect

    for ambiguous_ratio in ambiguous_ratios:
        ambiguous_patient_map = ontology_handler.build_ambiguous_patient_map(
            disease_ids=query,
            ambiguous_ratio=ambiguous_ratio,
        )

        for disease_id in ambiguous_patient_map.keys():
            assert len(ambiguous_patient_map[disease_id].pheno) != 0


def test_save_and_load_json(tmpdir, diseases, ontology_handler):
    target_path = tmpdir.mkdir("jsons").join("disease_map.json").strpath

    ontology_handler.save_to_json(source_dict=diseases, target_path=target_path)
    loaded_from_json = ontology_handler.load_from_json(json_path=target_path)

    for key in diseases.keys():
        assert diseases[key].disease_name == loaded_from_json[key].disease_name
        assert diseases[key].disease_id == loaded_from_json[key].disease_id
        assert diseases[key].pheno == loaded_from_json[key].pheno


@pytest.fixture(scope="function")
def real_world_case():
    real_world_case = {
        "EPJ21-PSFY": {
            "OMIM:619340": [
                "HP:0000559",
                "HP:0100689",
                "HP:0000615",
                "HP:0000613",
                "HP:0000518",
            ],
            "OMIM:619341": ["HP:0006554"],
        },
        "GPX22-SEVE": {"ORPHA:231160": ["HP:0000742", "HP:0001251"]},
        "EPG22-TKFN": {
            "3B:28592524:2017_Dec": [  # this kind of key does NOT exist in OMIM, obviously.
                "HP:0001629",
                "HP:0001655",
                "HP:0002817",
                "HP:0001643",
                "HP:0001642",
            ]
        },
        "EPJ22-HGGH": {"OMIM:619341": ["HP:0001999", "HP:0004313"]},
    }
    return real_world_case


def test_load_real_world_case(tmpdir, real_world_case, ontology_handler, diseases):
    real_world_tmp = tmpdir.mkdir("jsons").join("real_world_cases.json")
    with open(real_world_tmp.strpath, "a") as f:
        json.dump(real_world_case, f)

    ontology_handler.disease_pheno_map = diseases
    result = ontology_handler.load_real_world_case(json_path=real_world_tmp.strpath, n_patients=4)

    patient_ids = real_world_case.keys()
    for pid in patient_ids:
        if pid in result.keys():
            for disease_id, disease_obj in result[pid].diseases.items():
                assert disease_id in real_world_case[pid].keys()


def test_get_avg_pheno_num(ontology_handler, diseases):
    avg_phenos = ontology_handler.get_avg_pheno_num(patients=diseases)
    assert avg_phenos == 3.0


@pytest.mark.parametrize(
    "actual_patients",
    [
        pytest.param(
            {
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
                            pheno={
                                Phenotype(id="HP:0000559"),
                                Phenotype(id="HP:0100689"),
                                Phenotype(id="HP:0000615"),
                                Phenotype(id="HP:0000613"),
                                Phenotype(id="HP:0000518"),
                            },
                        ),
                    },
                ),
                "GPX22-SEVE": Patient(
                    id="GPX22-SEVE",
                    diseases={
                        "ORPHA:231160": Disease(
                            disease_id="ORPHA:231160",
                            disease_name="abcd",
                            pheno={
                                Phenotype(id="HP:0000742"),
                                Phenotype("HP:0001251"),
                            },
                        )
                    },
                ),
            }
        ),
    ],
)
def test_get_avg_pheno_num_real_world(ontology_handler, actual_patients):
    avg_phenos_real = ontology_handler.get_avg_pheno_num_real_world(patients=actual_patients)

    assert avg_phenos_real == pytest.approx(7 / 2, 1e-4)
