cat /dev/null > /tmp/nohup.out

#export DATASTORE_EMULATOR_HOST=localhost:8081
#export DATASTORE_EMULATOR_HOST=localhost:8200
#nohup dev_appserver.py app.yaml  --clear_datastore --host localhost --port 8100 > /tmp/nohup.out 2>&1 &
nohup dev_appserver.py app.yaml  --host localhost --port 9100 --admin_port=9000 > /tmp/fin_nohup.out 2>&1 &
