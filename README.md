## 개요
- 의미론적 유사도 계산기 평가를 위한 프로그램 [기존 논문 리뷰 참고](https://github.com/3billion/3ASC-Confirmed-variant-Resys/wiki/%EC%9D%98%EB%AF%B8%EB%A1%A0%EC%A0%81-%EC%9C%A0%EC%82%AC%EB%8F%84-%EC%B8%A1%EC%A0%95-%EB%B0%A9%EB%B2%95%EB%A1%A0:-%EB%B0%A9%EB%B2%95%EB%A1%A0-%EB%B0%8F-%ED%8E%98%EC%9D%B4%ED%8D%BC-%EB%AA%A8%EC%9D%8C)
- Optimal patients, Noisy patients 구현

## Prerequsite
```bash
$ pip install -r ../requirements.xtx

// Recommended
$ cd ../data
$ wget https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2023-01-27/hp.obo
```

## Patient's symptoms - Disease phenotype similiarty API
```bash
python3 SemanticSimiliarty -p {port} -w {n_workers}
```

## Request to API
```
curl -X 'POST' \
  'http://localhost:8000/calculate/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "symptoms": [
    "HP:0000123", "HP:0000001"
  ]
}'
```



## Build datasets
- build dataset from scratch 
```
from build_datasets import OntologyHandeler
ontology_handeler = OntologyHandeler(
        obo_file_path = "/data3/shared/cy-db-mirror/rawData/HPO/result/hp.obo",
        dataframe_path = "/data3/shared/cy-db-mirror/rawData/HPO/result/phenotype.hpoa",
        usecols = ["#DatabaseID", "DiseaseName", "HPO_ID"],
    ) #obo, hpoa 파일 경로가 default로

# build patients map
optimal_patients = ontology_handeler.build_optimal_patients_map() #사용자 정의 disease list를 전달할 수 있음
noisy_patients = ontology_handeler.build_noisy_patients_map() #사용자 정의 disease list를 전달할 수 있음

# save
ontology_handeler.save_to_json(source_dict=optimal_patients, target_path="somewhere to save")
```
- build dataset from json file
```
from build_datasets import OntologyHandeler
ontology_handeler = OntologyHandeler() #obo, hpoa 파일 경로가 default로
patients = ontology_handeler.load_from_json(json_path="somewhere to load")
```
- 출력 예시
```
>>> optimal_patients
>>> {
        "OMIM:619341": Disease(
            disease_name="Developmental and epileptic encephalopathy 100",
            disease_id="OMIM:619341",
            pheno={
                Phenotype(
                    pheno_name="Small for gestational age", hpoid="HP:0001518"
                ),
                Phenotype(pheno_name="Tonic seizure", hpoid="HP:0032792"),
            },
        ),
        "ORPHA:231160": Disease(
            disease_name="Familial cerebral saccular aneurysm",
            disease_id="ORPHA:231160",
            pheno={
                Phenotype(pheno_name="Encephalomalacia", hpoid="HP:0040197"),
                Phenotype(pheno_name="Aortic dissection", hpoid="HP:0002647"),
                Phenotype(pheno_name="Intracranial hemorrhage", hpoid="HP:0002170"),
                Phenotype(pheno_name="Aortic root aneurysm", hpoid="HP:0002616"),
                Phenotype(pheno_name="Oculomotor nerve palsy", hpoid="HP:0012246"),
            },
        ),
    ...
    }
```



## 질병유사도 성능테스트
```/bin/bash
$ python3 run_testbed.py --p optimal --n 200
```
