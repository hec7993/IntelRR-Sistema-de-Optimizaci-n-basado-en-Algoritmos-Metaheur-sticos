# IntelRR-Sistema-de-Optimizaci-n-basado-en-Algoritmos-Metaheur-sticos
Hay muchas tareas y varias maquinas virtuales  Hay que decidir que tarea se ejecuta en que maquina para que:      1) todo termine lo mas rapido posible        (tiempo total de ejecucion (makespan))      2) se gaste la menor energia posible      3) ninguna maquina se pase de su memoria disponible (si se pasa, hay un castigo)

IntelRR — Optimización híbrida GA-PSO para asignación de recursos en la nube
Proyecto final de Algoritmos Metaheurísticos. Asigna 100 tareas a 12 VMs heterogéneas minimizando
tiempo de respuesta (makespan) y consumo energético, con restricción de memoria por penalización.
Archivos
intelrr.py — TODO el proyecto en un solo archivo: modelo, líneas base, GA, PSO, GA-PSO, experimentos, estadística y figuras.
build_reporte.py — genera IntelRR_Reporte_Tecnico.pdf desde resultados/.
resultados — figuras PNG, resultados.json, tabla_resultados.csv, log_analisis.txt.
Ejecución
bash
pip install -r requirements.txt
python intelrr.py            # ~1-2 min; genera resultados/
python build_reporte.py     # regenera el PDF (opcional)
python intelrr.py --rapido   # prueba rápida (~10 s)
Todas las semillas están fijas: los resultados son reproducibles.
Demostración funcional
Mostrar la instancia y la función objetivo (crear_instancia, fitness en intelrr.py).
Ejecutar python intelrr.py y mostrar resultados/fig1_convergencia.png, fig4_asignacion.png.
Explicar el híbrido: hibrido_ga_pso (diversidad → mutación adaptativa; atracción a pbest/gbest; reemplazo elitista).
Discutir resultados con honestidad: GA-PSO ≈ GA ≈ Voraz en esta instancia, PSO es claramente peor.
