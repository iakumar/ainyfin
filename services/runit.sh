cat /dev/null > /tmp/fin_nohup.out

#export DATASTORE_EMULATOR_HOST=localhost:8081
#export DATASTORE_EMULATOR_HOST=localhost:8200
#nohup dev_appserver.py app.yaml  --clear_datastore --host localhost --port 8100 > /tmp/nohup.out 2>&1 &

nohup /Users/rithuhegde/ainyfin/services/fin/bin/python /Users/rithuhegde/google-cloud-sdk/bin/dev_appserver.py app.yaml  --host localhost --port 9100 --admin_port=9000 > /tmp/fin_nohup.out 2>&1 &
