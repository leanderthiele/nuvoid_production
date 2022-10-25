How to get optuna/mysql to work on Tiger.
Still no idea if the server will be persistent and stable...

1. Setting up the server
$ mysqld --initialize-insecure --datadir=$CONDA_PREFIX/data
[the -insecure is probably not best practice but whatever]
$ mysql.server start
        check status with `$ mysql.server status`
        stop with `$ mysql.server stop`

2. Creating a database and adding non-root user
$ mysql -u root
mysql> CREATE DATABASE `mydb`;
mysql> CREATE USER myuser IDENTIFIED BY 'mypwd';
mysql> GRANT ALL ON `mydb`.* TO myuser;
mysql> exit

3. Set non-default port to avoid potential conflicts with other users
edit the `port=` entries in .../etc/my.cnf to PORTNR

4. Using this database from a compute node
Use the following string as `storage` in optuna:
mysql://myuser:mypwd@tigercpu:PORTNR/mydb?unix_socket=/home/lthiele/mysql/mysql.sock

Note that one can have multiple studies in the same database,
which makes the entire thing much more convenient!
(they are identified by the `study_name`)

I have made the following environment for use within optuna:
mydb=optunadb
myser=optunausr
mypwd=pwd
PORTNR=3310
