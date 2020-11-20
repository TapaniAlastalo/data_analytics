const express = require('express')
const bodyParser= require('body-parser')
const MongoClient = require('mongodb').MongoClient
const app = express()

app.use(bodyParser.urlencoded({extended: true}))

var db, result

MongoClient.connect('mongodb://192.168.1.24:27017', { useNewUrlParser: true }, (err, client) => {
  if (err) return console.log(err)
  db = client.db('temperaturedatadb')
  app.listen(3030, () => {
    console.log('Palvelu käynnistetty porttiin 3030')
  })
})

app.get('/', (req, res) => {
    db.collection('temperaturedata').aggregate([
        { $group: { _id: "$_id", name: "$name", timestamp: "$timestamp", temp: "$temp", feelTemp: "$feel", minTemp: "$min", maxTemp: "$max" } }
    ]).toArray((err, result) => {
        if (err) return console.log(err)
        res.render('index.ejs', {temperaturedata: result})
    })
})

app.get('/reload', (req, res) => {
    result = []
    db.collection('temperaturedata').aggregate([
        { $group: { _id: "$_id", name: "$name", timestamp: "$timestamp", temp: "$temp", feelTemp: "$feel", minTemp: "$min", maxTemp: "$max" } }
    ]).toArray((err, result) => {
        if (err) return console.log(err)
        res.json(result)
    })
})

// date, time
