const videoElement = document.getElementsByClassName('input_video')[0];

const canvasElement = document.getElementsByClassName('output_canvas')[0];

const canvasCtx = canvasElement.getContext('2d');

const resultado = document.getElementById("resultado");

// -----------------------------
// LETRAS
// -----------------------------

const labels = ["A", "B", "C", "D", "E"];

// -----------------------------
// IA
// -----------------------------

let model;

// -----------------------------
// TEXTO
// -----------------------------

let texto = "";

let ultimaLetra = "";

let ultimoTiempo = Date.now();

const TIEMPO_ESPERA = 1500;

// Buffer
let buffer = [];

// -----------------------------
// CARGAR MODELO
// -----------------------------

async function cargarModelo() {

    model = await tf.loadLayersModel(
        '/static/modelo_web/model.json'
    );

    resultado.innerHTML = "IA cargada";
}

cargarModelo();

// -----------------------------
// HABLAR
// -----------------------------

function hablarTexto(textoHablar) {

    const speech = new SpeechSynthesisUtterance(textoHablar);

    speech.lang = "es-ES";

    speech.rate = 1;

    speechSynthesis.speak(speech);
}

// -----------------------------
// RESULTADOS
// -----------------------------

async function onResults(results) {

    canvasCtx.save();

    canvasCtx.clearRect(
        0,
        0,
        canvasElement.width,
        canvasElement.height
    );

    canvasCtx.drawImage(
        results.image,
        0,
        0,
        canvasElement.width,
        canvasElement.height
    );

    if (results.multiHandLandmarks) {

        let row = [];

        for (const landmarks of results.multiHandLandmarks) {

            // Dibujar
            drawConnectors(
                canvasCtx,
                landmarks,
                HAND_CONNECTIONS, {
                    color: '#00FF00',
                    lineWidth: 4
                }
            );

            drawLandmarks(
                canvasCtx,
                landmarks, {
                    color: '#FF0000',
                    lineWidth: 2
                }
            );

            // Coordenadas
            for (const lm of landmarks) {

                row.push(lm.x);
                row.push(lm.y);
                row.push(lm.z);
            }
        }

        // Completar
        while (row.length < 126) {

            row.push(0);
        }

        // -----------------------------
        // IA
        // -----------------------------

        if (model) {

            const tensor = tf.tensor([row]);

            const prediction = model.predict(tensor);

            const data = await prediction.data();

            let maxIndex = 0;

            for (let i = 1; i < data.length; i++) {

                if (data[i] > data[maxIndex]) {

                    maxIndex = i;
                }
            }

            const letra = labels[maxIndex];

            const confianza = data[maxIndex];

            // -----------------------------
            // BUFFER
            // -----------------------------

            if (confianza > 0.85) {

                buffer.push(letra);

                if (buffer.length > 10) {

                    buffer.shift();
                }
            }

            // -----------------------------
            // ESTABILIDAD
            // -----------------------------

            if (buffer.length === 10) {

                const conteo = {};

                buffer.forEach((l) => {

                    conteo[l] = (conteo[l] || 0) + 1;
                });

                let letraEstable = "";

                let maxConteo = 0;

                for (const letraBuffer in conteo) {

                    if (conteo[letraBuffer] > maxConteo) {

                        maxConteo = conteo[letraBuffer];

                        letraEstable = letraBuffer;
                    }
                }

                const tiempoActual = Date.now();

                if (
                    maxConteo >= 8 &&
                    letraEstable !== ultimaLetra &&
                    tiempoActual - ultimoTiempo > TIEMPO_ESPERA
                ) {

                    texto += letraEstable;

                    ultimaLetra = letraEstable;

                    ultimoTiempo = tiempoActual;

                    buffer = [];
                }
            }

            // -----------------------------
            // MOSTRAR
            // -----------------------------

            resultado.innerHTML = `
                Letra: ${letra}
                <br>
                Confianza: ${confianza.toFixed(2)}
                <br><br>
                Texto: ${texto}
            `;
        }
    }

    canvasCtx.restore();
}

// -----------------------------
// MEDIAPIPE
// -----------------------------

const hands = new Hands({

    locateFile: (file) => {

        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
});

hands.setOptions({

    maxNumHands: 2,

    modelComplexity: 1,

    minDetectionConfidence: 0.7,

    minTrackingConfidence: 0.7
});

hands.onResults(onResults);

// -----------------------------
// CAMARA
// -----------------------------

const camera = new Camera(videoElement, {

    onFrame: async() => {

        await hands.send({
            image: videoElement
        });
    },

    width: 1280,

    height: 720
});

camera.start();


// -----------------------------
// BOTON HABLAR MOVIL
// -----------------------------

const btnHablar = document.getElementById("btnHablar");

btnHablar.addEventListener("click", () => {

    if (texto.trim() !== "") {

        // Detener voz anterior
        speechSynthesis.cancel();

        // Crear voz
        const speech =
            new SpeechSynthesisUtterance(texto);

        // Configuracion
        speech.lang = "es-ES";

        speech.rate = 1;

        speech.volume = 1;

        speech.pitch = 1;

        // Hablar
        speechSynthesis.speak(speech);
    }
});


// -----------------------------
// TECLAS
// -----------------------------

document.addEventListener("keydown", (e) => {

    // Hablar
    if (e.key === "v") {

        hablarTexto(texto);
    }

    // Limpiar
    if (e.key === "c") {

        texto = "";
    }

    // Espacio
    if (e.key === " ") {

        texto += " ";
    }

    // Borrar
    if (e.key === "Backspace") {

        texto = texto.slice(0, -1);
    }
});