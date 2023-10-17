def get_db():
    """
    get mongodb database
    :param mdb_usr: mongodb user
    :param mdb_passw: mongodb password
    :return: db
    """
    import pymongo
    from conf.secrets_k import mdb_usr, mdb_passw
    if mdb_usr == 'xxx':
        print('debe configurar las credenciales de mongodb en fichero data/config.json.'
              'esta información se encuentra en el panel de back4app por ejemplo')
    cs = "mongodb+srv://" + mdb_usr + ":" + mdb_passw + "@cuentame.2tlxj.mongodb.net/"
    client = pymongo.MongoClient(cs)
    db = client.get_database('cuentame')
    print('** Collection names: ', db.list_collection_names())
    return db


def get_colls(db):
    """
    get collections from db
    :param db: mongodb database
    :return: col_libros, col_libros_sum
    """

    print('** Obteniendo colecciones de la base de datos')
    col_libros_sum = db.get_collection('librosSum')
    col_libros = db.get_collection('libros')
    print('Nro de libros (summaries)', col_libros_sum.count_documents({}))
    print('Ejemplo de un summary de libro: ', col_libros_sum.find_one())

    return col_libros, col_libros_sum


def borrar_objetos(id_borrar):
    print('se debe hacer algo como lo siguiente ')
    msg = f"""
from utils_mongo import get_db
db = utils_mongo.get_db()
a = db.libros.delete_many({{'nLibro': {id_borrar}}})
    """
    print(msg)
