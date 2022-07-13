const SendFile = (e) => {

    const files = e.target.files;
    $('#file-status').html(`${files[0].name}`);
    const formdata = new FormData();
    formdata.append("file", files[0]);

    fetch("http://localhost:5000/upload", {
            method: "POST",
            body: formdata,
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            $("#file-status").html(files[0].name + " " + data.message)
        });
};

const get_training_response = () => {
    console.log('training Function Clicked');
    $("#status").html("Training Started!");
    $("#complete").html("Please wait, training is in progress...");

    fetch("http://localhost:5000/train").then(response => response.json())
        .then(data => {
            $("#complete").html(data.message);
            if (data.message === "Forecasting - Done ✔ ") {
                $("#button").css({
                    'visibility': "visible"
                });
            }
        });
};

const get_forecast_data = () => {
    fetch("http://localhost:5000/forecast_data").then(response => response.json()).then(data => {

        let xvalues = Object.keys(data.message);

        xvalues = xvalues.map(ts => {
            date = new Date(parseInt(ts));
            return date.getDate() + "/" + (date.getMonth() + 1) + "/" + date.getFullYear()
        });
        const yvalues = Object.values(data.message);
        new Chart("myChart", {

            type: "line",
            data: {
                labels: xvalues,
                datasets: [{
                    label: 'Predicted Forecast',
                    data: yvalues
                }]
            },
            options: {
                responsive: true,
                scales: {
                    xAxes: [{
                        display: true,
                        scaleLabel: {
                            display: true,
                            labelString: 'Dates'
                        }
                    }],
                    yAxes: [{
                        display: true,
                        scaleLabel: {
                            display: true,
                            labelString: 'Production(Kgs)'
                        }
                    }]
                }
            }
        });
        $("#forecast-data").css({
            'visibility': "visible"
        });
    })
};