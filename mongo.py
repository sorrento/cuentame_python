def get_db(mdb_usr, mdb_passw):
    if mdb_usr =='xxx':
        print('debe configurar las credenciales de mongodb en fichero data/config.json.'
              'esta información se encuentra en el panel de back4app por ejemplo')
    cs = "mongodb+srv://" + mdb_usr + ":" + mdb_passw + "@cuentame.2tlxj.mongodb.net/"
    client = pymongo.MongoClient(cs)
    db = client.get_database('cuentame')
    print('**test', db.list_collection_names())
    return db


def get_colls(db):
    col_libros_sum = db.get_collection('librosSum')
    col_libros = db.get_collection('libros')
    print('test: nlibros sum (count)', col_libros_sum.count_documents({}))
    print('test libros, example:', col_libros_sum.find_one())

    return col_libros, col_libros_sum