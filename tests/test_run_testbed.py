import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock
from ..data_model import Disease, Phenotype, Patient, PatientId
from ..run_testbed import (
    SemanticSimilarityEvaluator,
    TopKListLenMissmatchException,
    get_prediction,
    get_prediction_real_world
)


@pytest.fixture(scope="function")
def patients():
    patients = {
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

    return patients


@pytest.fixture(scope="function")
def actual_patients():
    actual_patients = {
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

    return actual_patients


@pytest.fixture(scope="module")
def sorted_pred():
    sorted_pred = {
        "OMIM:619340": {
            "OMIM:619340": 0.99,
            "OMIM:619344": 0.88,
            "OMIM:619345": 0.77,
            "OMIM:619346": 0.66,
            "OMIM:619347": 0.55,
            "OMIM:619348": 0.44,
            "OMIM:619349": 0.33,
            "OMIM:619350": 0.22,
            "OMIM:619341": 0.11,
            "OMIM:619343": 0.01,
        },
        "OMIM:619341": {
            "OMIM:619345": 0.99,
            "OMIM:619346": 0.88,
            "OMIM:619347": 0.77,
            "OMIM:619340": 0.66,
            "OMIM:619341": 0.55,
            "OMIM:619350": 0.44,
            "OMIM:619344": 0.33,
            "OMIM:619343": 0.22,
            "OMIM:619348": 0.11,
            "OMIM:619349": 0.01,
        },
        "ORPHA:231160": {
            "OMIM:619340": 0.99,
            "OMIM:619345": 0.88,
            "OMIM:619346": 0.77,
            "OMIM:619349": 0.66,
            "OMIM:619341": 0.55,
            "OMIM:619343": 0.44,
            "OMIM:619344": 0.33,
            "OMIM:619347": 0.22,
            "OMIM:619348": 0.11,
            "ORPHA:231160": 0.01,
        },
    }
    return sorted_pred


@pytest.fixture(scope="function")
def sorted_pred_real():
    sorted_pred_real = {
        "EPJ21-PSFY": {
            "OMIM:619340": {
                "OMIM:619340": 0.99,
                "OMIM:619344": 0.88,
                "OMIM:619345": 0.77,
                "OMIM:619346": 0.66,
                "OMIM:619347": 0.55,
                "OMIM:619348": 0.44,
                "OMIM:619349": 0.33,
                "OMIM:619350": 0.22,
                "OMIM:619341": 0.11,
                "OMIM:619343": 0.01,
            },
            "OMIM:619341": {
                "OMIM:619345": 0.99,
                "OMIM:619346": 0.88,
                "OMIM:619347": 0.77,
                "OMIM:619340": 0.66,
                "OMIM:619341": 0.55,
                "OMIM:619350": 0.44,
                "OMIM:619344": 0.33,
                "OMIM:619343": 0.22,
                "OMIM:619348": 0.11,
                "OMIM:619349": 0.01,
            },
        },
        "GPX22-SEVE": {
            "ORPHA:231160": {
                "OMIM:619340": 0.99,
                "OMIM:619345": 0.88,
                "OMIM:619346": 0.77,
                "OMIM:619349": 0.66,
                "OMIM:619341": 0.55,
                "OMIM:619343": 0.44,
                "OMIM:619344": 0.33,
                "OMIM:619347": 0.22,
                "OMIM:619348": 0.11,
                "ORPHA:231160": 0.01,
            },
        },
    }
    return sorted_pred_real


@pytest.fixture(scope="module")
def evaluator():
    return SemanticSimilarityEvaluator(MagicMock(), MagicMock())


def test_get_prediction(patients, sorted_pred):
    calculator = Mock()
    calculator.get_disease_similarty.return_value = sorted_pred["OMIM:619340"]
    predictions = get_prediction(patients_map=patients, calculator=calculator)
    assert predictions["OMIM:619340"] == sorted_pred["OMIM:619340"]

def test_get_prediction_real_world(actual_patients, sorted_pred_real):
    calculator = Mock()
    calculator.get_disease_similarty.return_value = sorted_pred_real["EPJ21-PSFY"]["OMIM:619340"]
    predictions = get_prediction_real_world(patients_map=actual_patients, calculator=calculator)
    assert predictions["EPJ21-PSFY"]["OMIM:619340"] == sorted_pred_real["EPJ21-PSFY"]["OMIM:619340"]

@pytest.mark.parametrize(
    "top_k,score",
    [
        pytest.param(
            1,
            1 / 3,
            id="success1",
        ),
        pytest.param(
            5,
            2 / 3,
            id="success2",
        ),
        pytest.param(
            10,
            1.0,
            id="success3",
        ),
    ],
)
def test_get_top_k_recall(evaluator, patients, sorted_pred, top_k, score):
    accuracy = evaluator.get_top_k_recall(
        patients_map=patients,
        predicted=sorted_pred,
        top_k=top_k,
    )
    assert score == pytest.approx(accuracy, 1e-3)

@pytest.mark.parametrize(
    "top_k,score",
    [
        pytest.param(
            1,
            1 / 3,
            id="success1",
        ),
        pytest.param(
            5,
            2 / 3,
            id="success2",
        ),
        pytest.param(
            10,
            1.0,
            id="success3",
        ),
    ],
)
def test_get_top_k_recall_real_world(evaluator, actual_patients, sorted_pred_real, top_k, score):
    accuracy = evaluator.get_top_k_recall_real_world(
        patients_map=actual_patients,
        predicted=sorted_pred_real,
        top_k=top_k,
    )
    assert score == pytest.approx(accuracy, 1e-3)

def test_draw_top_k_plot_success(evaluator, tmpdir):
    save_path = Path(tmpdir.mkdir("figures").join("top-k_acc.png").strpath)
    assert not os.path.isfile(save_path)
    assert not save_path.exists()

    evaluator.draw_top_k_plot(
        top_ks=[1, 5, 10, 20, 30, 50, 100],
        accuracies=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98],
        file_path=save_path,
    )

    assert os.path.isfile(save_path)


def test_draw_top_k_plot_fail(evaluator, tmpdir):
    save_path = tmpdir.mkdir("figures").join("top-k_acc.png").strpath
    assert not os.path.isfile(save_path)

    with pytest.raises(TopKListLenMissmatchException):
        evaluator.draw_top_k_plot(
            top_ks=[
                1,
                5,
                10,
                20,
                30,
            ],
            accuracies=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98],
            file_path=save_path,
        )
