const API_URL = "https://sepsis-api.onrender.com/predict";

document.getElementById("predictionForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const data = {
        Hour: Number(Hour.value),
        HR: Number(HR.value),
        O2Sat: Number(O2Sat.value),
        SBP: Number(SBP.value),
        MAP: Number(MAP.value),
        DBP: Number(DBP.value),
        Resp: Number(Resp.value),
        Age: Number(Age.value),
        Gender: Number(Gender.value),
        Unit1: Number(Unit1.value),
        Unit2: Number(Unit2.value),
        HospAdmTime: Number(HospAdmTime.value),
        ICULOS: Number(ICULOS.value)
    };

    const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const json = await response.json();

    document.getElementById("resultBox").classList.remove("hidden");
    document.getElementById("prob").innerHTML = `<b>Probability:</b> ${json.probability}`;
    document.getElementById("label").innerHTML = `<b>Predicted Label:</b> ${json.predicted_label}`;
    document.getElementById("threshold").innerHTML = `<b>Threshold:</b> ${json.threshold_used}`;
});
