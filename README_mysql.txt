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

NOTE one may have to increase the max_connections variable for larger
     jobs, this can be done in the [mysqld] section in my.cnf

PERFORMANCE
I have done a test with 96 separate optuna processes, where the
objective evaluation takes ~1 sec.
This results in appreciable load on the head node SQL process.
(a lot of threads with low CPU utilization).
When I increase the objective evaluation time to 20 sec, we still
have a lot of head node SQL processes but they are idle -- as expected.
Hopefully sysadmins won't hate us for this...

---- CLEANING OPTUNA DB ----
This is sometimes necessary to delete trials which are spuriously
recorded as 'RUNNING'. The tricky thing is that there are a bunch
of tables with a foreign key constraint on the trial_id column into
the trials table, with the default referential action set up such that
one can't delete stuff. In order to clean, do the following:

[pseudocode!]

$ mysql -u root
mysql> use optunadb

! The following may not be required again as I changed our global
optunadb already, so perhaps it's just for reference.
for table_name in [heartbeats,
                   intermediate_values,
                   params,
                   system_attributes,
                   user_attributes,
                   values] do
        mysql> alter table trial_${table_name} drop foreign key trial_${table_name}_ibfk_1;
        mysql> alter table trial_${table_name} add constraint trial_${table_name}_ibfk_1 foreign key (trial_id) references trials (trial_id) on delete cascade;
done

! Make sure no optuna processes are actually running, otherwise this
will likely break some stuff...
for bad_state in [RUNNING, FAIL] do
        mysql> delete from trials where state='${bad_state}'
