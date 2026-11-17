from datetime import date, datetime
from unittest import mock

import pytest

from src.producto.notificar import ConfigCorreo, enviar, leer_config, resumen

COMPLETO = {
    "SMTP_SERVIDOR": "smtp.resend.com",
    "SMTP_PUERTO": "465",
    "SMTP_USUARIO": "resend",
    "SMTP_CLAVE_APP": "re_123",
    "CORREO_REMITENTE": "radar@albran.gt",
    "CORREO_DESTINO": "a@albran.gt, b@albran.gt",
}


def test_leer_config_completa_y_faltante():
    config = leer_config(COMPLETO)
    assert config.puerto == 465 and config.destinos == ["a@albran.gt", "b@albran.gt"]
    with pytest.raises(ValueError, match="SMTP_CLAVE_APP"):
        leer_config({**COMPLETO, "SMTP_CLAVE_APP": ""})


def test_resumen_con_lista_y_sin_lista(con):
    con.execute(
        "INSERT INTO procesos (ocid, nog, titulo, entidad, fecha_publicacion, fecha_cierre) "
        "VALUES ('a', '123', 'COMPRA DE COMPUTADORAS PARA EL IGSS', 'MINISTERIO DE SALUD', ?, ?)",
        [datetime(2026, 10, 9, 8), datetime(2026, 10, 11, 10)],
    )
    con.execute(
        "INSERT INTO predicciones (ocid, fecha_corrida, proyecto_similar, terminos_top, rango, "
        "seccion) VALUES ('a', '2026-10-09', 'Venta de equipo de cómputo', "
        "'a favor: computadora · en contra: —', 1, 'principal')"
    )
    asunto, cuerpo, texto = resumen(con, date(2026, 10, 9))
    assert asunto.startswith("Radar de Licitaciones: 1 concursos")
    assert "Compra de computadoras para el IGSS" in cuerpo and "nog=123" in cuerpo
    assert "2 días para ofertar" in texto
    asunto_vacio, _, _ = resumen(con, date(2026, 10, 8))
    assert "sin concursos relevantes" in asunto_vacio


@pytest.mark.parametrize("puerto, clase", [(465, "SMTP_SSL"), (587, "SMTP")])
def test_enviar_usa_ssl_o_starttls(puerto, clase):
    config = ConfigCorreo("smtp.resend.com", puerto, "resend", "re_1", "r@x.gt", ["d@x.gt"])
    with mock.patch(f"smtplib.{clase}") as smtp:
        enviar(config, "Asunto", "<p>hola</p>", "hola")
    sesion = smtp.return_value.__enter__.return_value
    sesion.login.assert_called_once_with("resend", "re_1")
    mensaje = sesion.send_message.call_args[0][0]
    assert mensaje["To"] == "d@x.gt" and mensaje["Subject"] == "Asunto"
    assert sesion.starttls.called == (puerto != 465)
