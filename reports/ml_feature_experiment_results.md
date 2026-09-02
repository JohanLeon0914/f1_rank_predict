# ML Feature Experiment Results

Fecha: 2026-09-02

## Objetivo

Probar nuevas features disponibles antes de la carrera y compararlas contra el modelo original. El criterio fue mantener el split temporal existente y evitar data leakage.

## Baseline original

Configuracion original de `XGBRanker`: `n_estimators=300`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.85`, `colsample_bytree=0.85`.

| Split | NDCG | Spearman | Kendall tau | MAE posicion | Top 1 | Top 3 | Top 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.9470 | 0.6490 | 0.5234 | 3.8090 | 0.5057 | 0.6475 | 0.7615 |
| test | 0.9478 | 0.6510 | 0.5253 | 3.3908 | 0.5200 | 0.6762 | 0.7743 |

## Features probadas

- Relativas pre-carrera: ranking de mejor tiempo de qualy, delta grid vs qualy, posiciones de penalizacion y gap porcentual a pole.
- Experiencia historica: conteos de carreras por piloto/constructor y en circuito.
- Tendencias recientes: diferencia entre forma de ultimas 3 carreras y ultimas 10.
- Fuerza relativa de temporada: puntos acumulados del piloto/constructor normalizados contra los rivales de la carrera.

## Resultado de ablation

| Variante | Val NDCG | Val Spearman | Val MAE | Val Top 1 | Val Top 3 | Test NDCG | Test Spearman | Test MAE | Test Top 1 | Test Top 3 | Test Top 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline recompuesto | 0.9470 | 0.6490 | 3.8090 | 0.5057 | 0.6475 | 0.9478 | 0.6510 | 3.3908 | 0.5200 | 0.6762 | 0.7743 |
| Relativas pre-carrera | 0.9473 | 0.6518 | 3.7767 | 0.4713 | 0.6456 | 0.9483 | 0.6551 | 3.3784 | 0.5257 | 0.6686 | 0.7806 |
| Experiencia historica | 0.9454 | 0.6487 | 3.8002 | 0.4540 | 0.6379 | 0.9499 | 0.6555 | 3.3808 | 0.5257 | 0.6762 | 0.7800 |
| Tendencias recientes | 0.9473 | 0.6484 | 3.8135 | 0.4943 | 0.6322 | 0.9494 | 0.6527 | 3.4073 | 0.4971 | 0.6743 | 0.7771 |
| Fuerza relativa temporada | 0.9466 | 0.6485 | 3.8046 | 0.5000 | 0.6475 | 0.9494 | 0.6506 | 3.4054 | 0.5257 | 0.6686 | 0.7766 |
| Todas las nuevas features | 0.9472 | 0.6490 | 3.8239 | 0.4655 | 0.6475 | 0.9500 | 0.6565 | 3.3748 | 0.5200 | 0.6800 | 0.7771 |

## Decision

Las nuevas features no se incorporaron porque la mejora en test fue pequena y el comportamiento en validation fue mixto, especialmente en `Top 1`.

La mejora aceptada fue ajustar hiperparametros manteniendo las features originales: `n_estimators=250`, `learning_rate=0.06`, `max_depth=3`, `subsample=0.90`, `colsample_bytree=0.90`.

| Modelo final | Val NDCG | Val Spearman | Val MAE | Val Top 1 | Val Top 3 | Test NDCG | Test Spearman | Test MAE | Test Top 1 | Test Top 3 | Test Top 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Features originales + tuning | 0.9469 | 0.6520 | 3.7851 | 0.5115 | 0.6533 | 0.9500 | 0.6535 | 3.3845 | 0.5657 | 0.6705 | 0.7760 |

