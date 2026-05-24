document.addEventListener("keydown", function (e) {
    if (e.key === "ArrowLeft") {
        document.getElementById("prev-btn").click();
    } else if (e.key === "ArrowRight") {
        document.getElementById("next-btn").click();
    }
});
