# Modelo Operativo para Pruebas de IA Seguras y Responsables (basado en OWASP AI Testing Guide)

## 1. Justificación Ejecutiva

**¿Quién prueba a los que deciden?**

Cuando una IA decide a quién darle crédito, predecir reincidencia criminal o priorizar pacientes, no estamos ante software tradicional: estamos frente a sistemas que *construyen realidad* mediante inferencias opacas y entrenamientos invisibles. Sin embargo, la mayoría de las organizaciones siguen testeando estos modelos como si fueran calculadoras estadísticas. Grave error.

Este modelo operativo propone una ruptura necesaria: transformar el testing de IA en un **proceso estructurado de validación social, técnica, ética y regulatoria**, tan riguroso como el despliegue de un medicamento o un sistema financiero.

Inspirado en la OWASP AI Testing Guide, pero extendido con experiencia de campo, este documento no se conforma con detectar vulnerabilidades. Busca establecer un nuevo estándar operativo para equipos que diseñan, auditan, aprueban o supervisan algoritmos que afectan derechos, decisiones y reputación organizacional.

## 2. Definiciones Clave

- **Bias (Sesgo):** Desviación sistemática que favorece o desfavorece injustamente a ciertos grupos.
- **Robustez adversaria:** Capacidad del modelo de mantener su desempeño ante inputs diseñados maliciosamente.
- **Privacidad diferencial (ε):** Medida formal del nivel de privacidad preservado por un algoritmo.
- **Data Drift:** Cambio en la distribución estadística de los datos de entrada respecto a los de entrenamiento.
- **OOD (Out-of-Distribution):** Datos que no pertenecen al dominio original del modelo, potencialmente peligrosos.

## 3. Marcos Regulatorios de Referencia

- **Ley N° 19.628 (Chile):** Sobre protección de la vida privada y datos personales. Base legal nacional que rige el tratamiento de datos en sistemas automatizados, aplicable a IA que utilice o procese datos personales.
- **Ley Marco de Ciberseguridad (en proceso):** Propuesta legislativa que establece obligaciones específicas en materia de infraestructura crítica y seguridad digital.
- **AI Act (Unión Europea):** Referencia internacional para clasificación de riesgos, documentación técnica y gobernanza algorítmica.
- **GDPR:** Estándar europeo sobre privacidad que establece derechos como la explicación de decisiones automatizadas.
- **ISO/IEC 23894:** Norma para gestión de riesgos en IA.
- **OECD AI Principles:** Recomendaciones para promover IA confiable, centrada en el ser humano.

## 4. Arquitectura del Modelo Operativo

El modelo propuesto se compone de cinco capas integradas:

1. **Gobernanza Estratégica**: Comité de IA, políticas corporativas, mapa de riesgos.
2. **Ciclo de Desarrollo Seguro**: Validaciones incorporadas desde la fase de diseño (shift-left).
3. **Pruebas Especializadas de AI**: Fairness, adversarial, privacidad, robustez, trazabilidad.
4. **Monitoreo y Evaluación Continua**: Métricas automáticas, detección de desviaciones, auditorías.
5. **Control Documental y Trazabilidad**: Model Cards, Datasheets, Registro de Validación.

## 5. Procesos Operativos por Etapa

### Etapa 1: Diseño y Planificación
- Definir caso de uso y nivel de criticidad.
- Elaborar matriz de riesgos y plan de testing alineado a AI Act.
- Establecer roles de revisión técnica y ética.

### Etapa 2: Desarrollo
- Aplicar testing de fairness con herramientas (Fairlearn, AIF360).
- Evaluar explicabilidad (SHAP, LIME).
- Incluir defensas adversarias y verificación contra prompt injection (si LLM).

### Etapa 3: Validación
- Ejecutar batería de pruebas técnicas (adversarios, privacidad, precisión).
- Generar informe de validación firmado por QA, legal, comité ético.
- Aprobar solo si el cumplimiento supera el umbral establecido (ej. 90%).

### Etapa 4: Despliegue y Monitoreo
- Despliegue controlado con logging avanzado.
- Activar dashboards con métricas clave (fairness, accuracy, drift).
- Alertas automatizadas por deterioro o sesgo emergente.

### Etapa 5: Auditoría y Mejora Continua
- Auditoría mensual cruzada por equipos distintos.
- Retroalimentación automática al pipeline de entrenamiento.
- Revisión anual por comité ampliado.

## 6. Herramientas por Dominio de Prueba

| Dominio          | Herramientas sugeridas                    |
|------------------|--------------------------------------------|
| Equidad          | AIF360, Fairlearn, What-If Tool            |
| Robustez         | ART, CleverHans, RobustBench               |
| Privacidad       | Diffprivlib, TensorFlow Privacy            |
| Interpretabilidad| SHAP, LIME, Captum                         |
| Trazabilidad     | Model Cards, Datasheets for Datasets       |
| MLOps/Monitoreo  | Arize AI, Evidently.ai, MLflow             |

## 7. Métricas y KPIs Profesionales

- **% de cobertura de pruebas por modelo**
- **Tasa de falsos positivos/negativos por grupo demográfico**
- **Índice de robustez adversaria** (ataques exitosos / intentos)
- **Tiempo medio de reentrenamiento tras detección de drift**
- **Nivel de cumplimiento del plan de validación (%)**
- **% de modelos con documentación completa (card + datasheet)**

## 8. Flujo de Trabajo Institucional (Texto Representativo)

1. PM de IA propone nuevo modelo al Comité de Gobernanza.
2. Se activa plan de pruebas y checklist de validación OWASP-AI.
3. Desarrollo incluye defensas, logging, validación fairness.
4. QA y Compliance revisan y firman validación.
5. Comité aprueba despliegue. Se activa monitoreo.
6. Alertas activan revisión o rollback automatizado.
7. Revisión post-mortem y actualización del modelo operativo.

## 9. Anexos Sugeridos

- **Template de Plan de Validación Técnica**
- **Matriz de Riesgos por Tipo de IA (LLM, visión, tabular)**
- **Formulario de Registro de Evaluación Ética**
- **Checklist de Validación OWASP-AI por Dominio**

## 10. Conclusión

Una guía profesional debe ser ejecutable, auditada y adaptable. El presente modelo ofrece un marco riguroso, pero operativo, que permite alinear prácticas de desarrollo, cumplimiento normativo y control estratégico en el despliegue de sistemas de IA. Requiere adopción transversal, gobernanza técnica y cultura organizacional madura. Aporta trazabilidad, prevención de riesgos y legitimidad social al uso de algoritmos en contextos reales.

