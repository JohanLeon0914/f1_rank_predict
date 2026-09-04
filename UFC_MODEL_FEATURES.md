# UFC Winner Model: Features, Entrenamiento y Resultados

## Objetivo

El modelo predice el ganador de una pelea UFC como clasificacion binaria:

- `red_win = 1`: gana el peleador enviado como `red_fighter_id`.
- `red_win = 0`: gana el peleador enviado como `blue_fighter_id`.

El modelo final es `XGBClassifier` con `objective="binary:logistic"`.

## Datos usados

CSV principales:

- `UFC/master.csv`: peleas, evento, fecha, ganador, metodo y estadisticas agregadas de cada pelea.
- `UFC/fighter.csv`: datos de perfil del peleador.

Se entreno con 11.238 peleas validas:

- Se excluyeron `draw`, `no_contest`, peleas sin ganador y filas donde `winner_id` no correspondia a ninguno de los dos peleadores.
- El dataset se ordeno por `event_date`.
- Cada fila se construyo antes de actualizar los historiales con la pelea actual, para evitar usar informacion del futuro.

## Prevencion de data leakage

Las estadisticas de rendimiento de `fighter.csv` como `slpm`, `str_acc`, `sapm`, `str_def`, `td_avg`, `td_acc`, `td_def` y `sub_avg` parecen ser agregados actuales del perfil del peleador. Por eso no se usaron como features estaticas de entrenamiento: podrian incluir peleas futuras cuando el modelo aprende peleas antiguas.

El modelo usa como rendimiento real los acumulados calculados desde `master.csv` usando solo peleas anteriores a la fecha objetivo.

Features estaticas permitidas:

- altura en pulgadas;
- peso en libras;
- alcance en pulgadas;
- edad calculada en la fecha de pelea;
- stance;
- peso pactado de la pelea;
- indicador de pelea titular.

## Grupos de features

### Perfil fisico y contexto

Por cada peleador:

- `red_height_inches`, `blue_height_inches`
- `red_weight_lbs`, `blue_weight_lbs`
- `red_reach_inches`, `blue_reach_inches`
- `red_age_years`, `blue_age_years`
- `red_stance`, `blue_stance`
- `stance_matchup`

Contexto:

- `weight_class`
- `title_fight`

Diferencias:

- `diff_height_inches`
- `diff_weight_lbs`
- `diff_reach_inches`
- `diff_age_years`

### Historial de carrera

Por cada peleador:

- peleas, victorias y derrotas acumuladas;
- win rate historico;
- tasa de victoria por KO/TKO;
- tasa de victoria por submission;
- tasa de victoria por decision;
- tasa de derrotas por finalizacion;
- promedios historicos de knockdowns;
- golpes significativos conectados e intentados;
- golpes totales conectados e intentados;
- derribos exitosos e intentados;
- intentos de submission;
- reversals;
- tiempo de control;
- rounds peleados;
- duracion estimada de pelea;
- accuracy historica de golpeo significativo;
- accuracy historica de derribo;
- defensa aproximada via estadisticas recibidas.

### Ultimas 3, 5 y 10 peleas

Para ventanas `last_3`, `last_5` y `last_10`, por cada peleador:

- cantidad de peleas disponibles en la ventana;
- promedio de victoria;
- promedios de KO/TKO win/loss, submission win/loss, decision win/loss;
- promedio de finish win/loss;
- knockdowns a favor y en contra;
- golpes significativos a favor y en contra;
- golpes totales a favor y en contra;
- derribos a favor y en contra;
- intentos de submission a favor y en contra;
- reversals a favor y en contra;
- control a favor y en contra;
- rounds y duracion promedio;
- accuracy de golpeo significativo;
- accuracy de derribo.

### Head-to-head

Para el par historico entre ambos peleadores:

- `h2h_total_fights`
- `h2h_red_wins`
- `h2h_blue_wins`
- `h2h_red_win_rate`
- `h2h_win_diff`

### Diferenciales dinamicos

Para las features historicas principales se genera tambien:

- `diff_*`

Ejemplo:

- `diff_career_win_rate`
- `diff_last_10_avg_win`
- `diff_last_5_td_acc_for`
- `diff_last_3_avg_ctrl_seconds_for`

## Entrenamiento

Comando:

```bash
.venv-linux/bin/python scripts/train_ufc_model.py
```

Split temporal:

- Train: 7.866 peleas, desde 1993-11-12 hasta 2020-10-03.
- Validation: 1.686 peleas, desde 2020-10-03 hasta 2023-09-05.
- Test: 1.686 peleas, desde 2023-09-05 hasta 2026-08-08.

El split es temporal para simular predicciones futuras y reducir fuga de informacion.

## Resultados

Validation:

- Accuracy: 0.5985
- ROC AUC: 0.6268
- Log loss: 0.6632
- Brier score: 0.2350
- Precision: 0.6149
- Recall: 0.7967
- F1: 0.6941

Test:

- Accuracy: 0.6115
- ROC AUC: 0.6463
- Log loss: 0.6605
- Brier score: 0.2330
- Precision: 0.6151
- Recall: 0.7964
- F1: 0.6941

Matriz de confusion test:

- Blue real / Blue predicho: 288
- Blue real / Red predicho: 465
- Red real / Blue predicho: 190
- Red real / Red predicho: 743

## Feature importance

Top 20 del entrenamiento actual:

1. `red_last_10_avg_sig_landed_against`
2. `red_career_avg_sig_landed_against`
3. `red_last_3_avg_ctrl_seconds_against`
4. `blue_last_10_avg_sig_landed_for`
5. `blue_reach_inches`
6. `blue_career_avg_sig_attempted_for`
7. `diff_age_years`
8. `red_last_10_avg_win`
9. `diff_last_3_avg_ctrl_seconds_for`
10. `red_career_avg_ctrl_seconds_against`
11. `red_career_win_rate`
12. `blue_last_10_avg_sig_attempted_for`
13. `blue_last_3_avg_sig_attempted_for`
14. `blue_last_3_avg_sig_landed_for`
15. `blue_last_10_avg_td_attempted_for`
16. `red_last_10_avg_ctrl_seconds_for`
17. `diff_last_10_avg_win`
18. `red_last_5_avg_ctrl_seconds_against`
19. `red_last_10_avg_finish_win`
20. `red_last_10_avg_ctrl_seconds_against`

Archivos generados:

- CSV completo: `models/ufc/ufc_feature_importance.csv`
- Grafica SVG: `reports/ufc/ufc_feature_importance.svg`
- Curva ROC SVG: `reports/ufc/ufc_roc_curve.svg`
- Metricas JSON: `reports/ufc/ufc_metrics.json`

## Inferencia

Para una pelea futura:

1. Se carga el modelo y el preprocessor.
2. Se determina `fight_date`. Si no se envia, se usa el dia posterior al ultimo evento historico.
3. Se reconstruye el historial de ambos peleadores usando solo peleas con `event_date < fight_date`.
4. Se infiere `weight_class` si no fue enviada.
5. Se calculan features fisicas, career, ultimas 3/5/10, head-to-head y diferenciales.
6. El modelo devuelve `red_win_probability`.
7. La probabilidad azul se calcula como `1 - red_win_probability`.
8. El endpoint devuelve estadisticas historicas y contribuciones explicativas.

## Limitaciones

- El modelo no conoce lesiones, campamento, short notice, odds, pesaje real, cambios tacticos ni informacion externa a los CSV.
- Algunos peleadores tienen poco historial, por lo que varias features quedan faltantes y se imputan con medianas del entrenamiento.
- El orden `red`/`blue` puede influir porque el modelo aprende desde esa perspectiva. Para aplicaciones criticas se recomienda evaluar tambien una version simetrizada o entrenar con filas duplicadas intercambiando esquinas.
- Las probabilidades no son garantias y no deben presentarse como consejo de apuestas.

