#!/usr/bin/python3
# -*- coding: UTF-8 -*-

from statemachine import StateMachine, State
import os
from ardSerial import *
from picamera import PiCamera
from time import sleep
from vgg16_D2 import inference_d2

# Definición directorio de trabajo
ROOT_DIR = '/home/gabriel/Desktop/TFM'
DATA_INFERENCE = os.path.join(ROOT_DIR, "INFERENCE")

# Inicialización cámara
camera = PiCamera()
camera.rotation = -90
sleep(2)

# Variable de control
control = True

# Configuración Bittle
model = 'Bittle'
postureTable = postureDict[model]

E_RGB_ALL = 0
E_RGB_RIGHT = 1
E_RGB_LEFT = 2

E_EFFECT_BREATHING = 0
E_EFFECT_ROTATE = 1
E_EFFECT_FLASH = 2
E_EFFECT_NONE = 3

# Instrucciones de movimiento
sit = ['ksit',1]
gira_right = ['kcrR',1]
gira_left = ['kcrL',1]
go = ['kcrF', 1]

# Definición de la máquina de estados
class D_2_Machine(StateMachine):
    
    # Estados del sistema
    inicio = State('INICIO', initial=True)
    destino = State('DESTINO')
    busca_left = State('Busca_left')
    busca_right = State('Busca_right')
    busca_center = State('Busca_center')

    # Transiciones del sistema
    start_start = inicio.to(inicio)
    start_right = inicio.to(busca_right)
    start_left = inicio.to(busca_left)
    start_center = inicio.to(busca_center)
    start_destino = inicio.to(destino)

    right_right = busca_right.to(busca_right)
    right_left = busca_right.to(busca_left)
    right_destino = busca_right.to(destino)
    right_center = busca_right.to(busca_center)

    left_left = busca_left.to(busca_left)
    left_right = busca_left.to(busca_right)
    left_destino = busca_left.to(destino)
    left_center = busca_left.to(busca_center)
    
    center_left = busca_center.to(busca_left)
    center_right = busca_center.to(busca_right)
    center_center = busca_center.to(busca_center)
    center_destino = busca_center.to(destino)
    
    # Acción de control en estado inicial
    global levanta
    levanta = ['kzero',1]
    
    def on_enter_inicio(self):
        print('estado_inicio')
        send( goodPorts, levanta)

if __name__ == '__main__':
    try:
        
        # Conexión del puerto serie
        goodPorts = {}
        connectPort(goodPorts)
        t=threading.Thread(target = keepCheckingPort, args = (goodPorts,))
        t.start()
        parallel = False
        time.sleep(2);

        '''
        LAZO DE CONTROL
        '''
        # Cambiar el directorio de trabajo e inicializar la máquina de estados
        os.chdir(DATA_INFERENCE)
        control_mov = D_2_Machine()
        control_mov.start_start()
        
        while(control==True):
            
            # Capturar imagen
            camera.capture("foto_inf.jpg")
            # Predicción del modelo
            y_class = inference_d2()
            print(y_class)
            
            # Según la clasificación ejecutar una transición
            if control_mov.is_inicio and y_class == 0:
                control_mov.start_center()
            elif control_mov.is_inicio and y_class == 1:
                control_mov.start_destino()
            elif control_mov.is_inicio and y_class == 2:
                control_mov.start_left()
            elif control_mov.is_inicio and y_class == 3:
                control_mov.start_right()
                
            elif control_mov.is_busca_left and y_class == 0:
                control_mov.left_center()
            elif control_mov.is_busca_left and y_class == 1:
                control_mov.left_destino()
            elif control_mov.is_busca_left and y_class == 2:
                control_mov.left_left()
            elif control_mov.is_busca_left and y_class == 3:
                control_mov.left_right()
                
            elif control_mov.is_busca_right and y_class == 0:
                control_mov.right_center()
            elif control_mov.is_busca_right and y_class == 1:
                control_mov.right_destino()
            elif control_mov.is_busca_right and y_class == 2:
                control_mov.right_left()
            elif control_mov.is_busca_right and y_class == 3:
                control_mov.right_right()
                
            elif control_mov.is_busca_center and y_class == 0:
                control_mov.center_center()
            elif control_mov.is_busca_center and y_class == 1:
                control_mov.center_destino()
            elif control_mov.is_busca_center and y_class == 2:
                control_mov.center_left()
            elif control_mov.is_busca_center and y_class == 3:
                control_mov.center_right()
            
            # Acción de control en cada estado del sistema
            if control_mov.is_destino:
                print('estado destino')
                send(goodPorts, sit)
                control = False
                
            if control_mov.is_busca_left:
                print('estado left')
                send(goodPorts, gira_left)    
                    
            if control_mov.is_busca_right:
                print('estado right')
                send(goodPorts, gira_right)
                    
            if control_mov.is_busca_center:
                print('estado center')
                send(goodPorts, go)
                   
            # Instrucción de pausa al finalizar cada iteración
            send(goodPorts, "kbalance",1)
               
        # Cerrar el canal de comunicación
        closeAllSerial(goodPorts)
        logger.info("finish!")
        os._exit(0)

    # Excepción en caso de no activar el canal de comunicación
    except Exception as e:
        logger.info("Exception")
        closeAllSerial(goodPorts)
        os._exit(0)
        raise e
