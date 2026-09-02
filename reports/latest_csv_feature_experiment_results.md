# Latest CSV Feature Experiment Results

Fecha: 2026-09-02

## Objetivo

Revisar los CSV agregados o actualizados en `F1/CSVs`, detectar nuevas senales pre-carrera y reentrenar el modelo con los datos mas recientes.

## CSV revisados

- `drivers.csv`: edad y nacionalidad del piloto.
- `constructors.csv`: nacionalidad del constructor.
- `sprint_results.csv`: resultados de carreras sprint desde 2021.
- `status.csv`: catalogo descriptivo de `statusId`; no se uso porque el resultado/estado de carrera es post-carrera.
- `seasons.csv`: calendario por temporada; no aporta una senal adicional frente a `races.csv`.

## Baseline con CSV actuales

Mismo modelo y features anteriores, entrenado contra los CSV actualizados.

| Split | NDCG | Spearman | Kendall tau | MAE posicion | Top 1 | Top 3 | Top 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.9469 | 0.6520 | 0.5281 | 3.7851 | 0.5115 | 0.6533 | 0.7632 |
| test | 0.9498 | 0.6529 | 0.5259 | 3.3894 | 0.5632 | 0.6705 | 0.7759 |

## Features candidatas

- Features estaticas de piloto/constructor: edad del piloto, carrera local del piloto, carrera local del constructor y coincidencia de pais piloto-constructor.
- Historial sprint previo: starts, posicion media, puntos medios, tasas top 3/top 8/DNF y forma en los ultimos 3 sprints para piloto y constructor.

Las features sprint usan solo sprints de carreras anteriores, no el sprint de la misma carrera, para mantener compatibilidad con la inferencia actual.

## Ablation

| Variante | Val NDCG | Val Spearman | Val MAE | Val Top 1 | Val Top 3 | Test NDCG | Test Spearman | Test MAE | Test Top 1 | Test Top 3 | Test Top 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline sin nuevas features | 0.9469 | 0.6520 | 3.7851 | 0.5115 | 0.6533 | 0.9498 | 0.6529 | 3.3894 | 0.5632 | 0.6705 | 0.7759 |
| Estaticas piloto/constructor | 0.9465 | 0.6502 | 3.8047 | 0.4943 | 0.6418 | 0.9498 | 0.6528 | 3.3853 | 0.5460 | 0.6820 | 0.7782 |
| Historial sprint previo | 0.9473 | 0.6531 | 3.7870 | 0.5172 | 0.6437 | 0.9505 | 0.6565 | 3.3591 | 0.5517 | 0.6858 | 0.7787 |
| Todas las nuevas features | 0.9468 | 0.6510 | 3.7939 | 0.5000 | 0.6533 | 0.9499 | 0.6527 | 3.3786 | 0.5575 | 0.6667 | 0.7747 |

## Decision

Se incorporo solo el historial sprint previo. Mejora ranking general en test (`NDCG`, `Spearman`, `MAE`, `Top 3`, `Top 10`) y tambien sube `Top 1` en validation. No se incorporaron las features estaticas porque fueron mas ruidosas.

