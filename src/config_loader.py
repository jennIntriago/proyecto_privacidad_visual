"""
Utilidad para cargar archivos de configuración YAML 
"""
import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigLoader:
    """Clase para cargar y validar configuraciones de modelos"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Inicializa el cargador de configuraciones
        
        Args:
            config_dir: Directorio donde están los archivos de configuración
        """
        if config_dir is None:
            project_root = Path(__file__).parent.parent
            self.config_dir = project_root / "configs"
        else:
            self.config_dir = Path(config_dir)
        
        self.config_files = {
            'resnet50': 'resnet_config.yaml',
            'densenet121': 'densenet_config.yaml', 
            'yolov8': 'yolo_config.yaml'
        }
        
        self.model_info = {
            'resnet50': {
                'full_name': 'ResNet50',
                'type': 'CNN Transfer Learning',
                'framework': 'PyTorch',
                'task': 'Binary Classification'
            },
            'densenet121': {
                'full_name': 'DenseNet121',
                'type': 'Dense CNN with MixUp',
                'framework': 'PyTorch', 
                'task': 'Binary Classification'
            },
            'yolov8': {
                'full_name': 'YOLOv8 Classification',
                'type': 'YOLO Classification',
                'framework': 'Ultralytics',
                'task': 'Binary Classification'
            }
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Carga un archivo de configuración YAML
        
        Args:
            config_path: Ruta al archivo de configuración
            
        Returns:
            dict: Configuración cargada
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuración cargada exitosamente: {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Archivo de configuración no encontrado: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error al parsear YAML: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al cargar configuración: {e}")
            raise
    
    def get_config_path(self, model_name: str) -> Path:
        """
        Obtiene la ruta del archivo de configuración para un modelo específico
        
        Args:
            model_name: Nombre del modelo ('resnet50', 'densenet121', 'yolov8')
            
        Returns:
            Path: Ruta al archivo de configuración
        """
        if model_name not in self.config_files:
            available_models = list(self.config_files.keys())
            raise ValueError(f"Modelo no soportado: {model_name}. Opciones: {available_models}")
        
        config_path = self.config_dir / self.config_files[model_name]
        
        if not config_path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
        
        return config_path
    
    def load_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Carga la configuración para un modelo específico
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            dict: Configuración del modelo
        """
        config_path = self.get_config_path(model_name)
        config = self.load_config(config_path)
        
        # Validar configuración básica
        self.validate_config(config, model_name)
        
        return config
    
    def validate_config(self, config: Dict[str, Any], model_name: str) -> bool:
        """
        Valida que la configuración tenga los campos requeridos
        
        Args:
            config: Configuración a validar
            model_name: Nombre del modelo
            
        Returns:
            bool: True si la configuración es válida
        """
        # Secciones requeridas básicas
        required_sections = ['model', 'data', 'training']
        
        # Validaciones específicas por modelo
        model_specific_requirements = {
            'resnet50': {
                'sections': ['model.classifier', 'training.phase1', 'training.phase2'],
                'fields': ['model.num_classes', 'data.batch_size']
            },
            'densenet121': {
                'sections': ['regularization.mixup', 'loss'],
                'fields': ['model.num_classes', 'training.epochs', 'regularization.mixup.alpha']
            },
            'yolov8': {
                'sections': ['augmentation', 'evaluation'],
                'fields': ['model.variant', 'data.input_size', 'training.epochs']
            }
        }
        
        # Validar secciones básicas
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Sección requerida '{section}' no encontrada en configuración de {model_name}")
        
        # Validar campos específicos del modelo
        if model_name in model_specific_requirements:
            requirements = model_specific_requirements[model_name]
            
            # Validar secciones específicas
            for section_path in requirements.get('sections', []):
                self._validate_nested_field(config, section_path, model_name)
            
            # Validar campos específicos
            for field_path in requirements.get('fields', []):
                self._validate_nested_field(config, field_path, model_name)
        
        logger.info(f"Configuración de {model_name} validada exitosamente")
        return True
    
    def _validate_nested_field(self, config: Dict[str, Any], field_path: str, model_name: str):
        """Valida un campo anidado usando notación de puntos"""
        keys = field_path.split('.')
        current = config
        
        for key in keys:
            if key not in current:
                raise ValueError(f"Campo requerido '{field_path}' no encontrado en configuración de {model_name}")
            current = current[key]
    
    def get_available_models(self) -> List[str]:
        """
        Obtiene lista de modelos disponibles
        
        Returns:
            list: Lista de nombres de modelos disponibles
        """
        return list(self.config_files.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, str]:
        """
        Obtiene información detallada de un modelo
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            dict: Información del modelo
        """
        if model_name not in self.model_info:
            raise ValueError(f"Información no disponible para modelo: {model_name}")
        
        return self.model_info[model_name]
    
    def print_config_summary(self, model_name: str):
        """
        Imprime un resumen de la configuración del modelo
        
        Args:
            model_name: Nombre del modelo
        """
        try:
            config = self.load_model_config(model_name)
            model_info = self.get_model_info(model_name)
            
            print(f"\n{'='*60}")
            print(f"CONFIGURACIÓN DE {model_name.upper()}")
            print(f"{'='*60}")
            
            # Información básica del modelo
            print(f"📋 Información del Modelo:")
            print(f"   • Nombre completo: {model_info['full_name']}")
            print(f"   • Tipo: {model_info['type']}")
            print(f"   • Framework: {model_info['framework']}")
            print(f"   • Tarea: {model_info['task']}")
            
            # Información del proyecto
            project_info = config.get('project', {})
            print(f"\n🚀 Proyecto:")
            print(f"   • Nombre: {project_info.get('name', 'N/A')}")
            print(f"   • Descripción: {project_info.get('description', 'N/A')}")
            print(f"   • Versión: {project_info.get('version', 'N/A')}")
            
            # Configuración del modelo
            model_config = config.get('model', {})
            print(f"\n🏗️ Modelo:")
            print(f"   • Arquitectura: {model_config.get('architecture', 'N/A')}")
            print(f"   • Preentrenado: {model_config.get('pretrained', 'N/A')}")
            print(f"   • Clases: {model_config.get('num_classes', 'N/A')}")
            
            # Configuración específica por modelo
            if model_name == 'resnet50':
                self._print_resnet_specific(config)
            elif model_name == 'densenet121':
                self._print_densenet_specific(config)
            elif model_name == 'yolov8':
                self._print_yolo_specific(config)
            
            # Configuración de datos
            data_config = config.get('data', {})
            print(f"\n📊 Datos:")
            print(f"   • Tamaño de imagen: {data_config.get('input_size', 'N/A')}")
            print(f"   • Batch size: {data_config.get('batch_size', 'N/A')}")
            
            # Configuración de entrenamiento
            training_config = config.get('training', {})
            print(f"\n🏋️ Entrenamiento:")
            if model_name == 'resnet50':
                print(f"   • Épocas Fase 1: {training_config.get('phase1', {}).get('epochs', 'N/A')}")
                print(f"   • Épocas Fase 2: {training_config.get('phase2', {}).get('epochs', 'N/A')}")
            else:
                print(f"   • Épocas: {training_config.get('epochs', 'N/A')}")
            
            optimizer_config = training_config.get('optimizer', {})
            print(f"   • Optimizador: {optimizer_config.get('type', 'N/A')}")
            
            print(f"{'='*60}")
            
        except Exception as e:
            logger.error(f"Error al mostrar resumen de {model_name}: {e}")
            raise
    
    def _print_resnet_specific(self, config: Dict[str, Any]):
        """Imprime información específica de ResNet50"""
        print(f"\n🔧 Específico ResNet50:")
        loss_config = config.get('loss', {})
        print(f"   • Función de pérdida: {loss_config.get('type', 'N/A')}")
        print(f"   • Label smoothing: {loss_config.get('label_smoothing', 'N/A')}")
        
        metrics_config = config.get('metrics', {})
        print(f"   • Métrica principal: {metrics_config.get('primary', 'N/A')}")
    
    def _print_densenet_specific(self, config: Dict[str, Any]):
        """Imprime información específica de DenseNet121"""
        print(f"\n🔧 Específico DenseNet121:")
        
        mixup_config = config.get('regularization', {}).get('mixup', {})
        print(f"   • MixUp: {mixup_config.get('enabled', 'N/A')}")
        print(f"   • MixUp Alpha: {mixup_config.get('alpha', 'N/A')}")
        
        loss_config = config.get('loss', {})
        print(f"   • Focal Loss Gamma: {loss_config.get('gamma', 'N/A')}")
        
        freeze_config = config.get('model', {})
        print(f"   • Estrategia de congelamiento: {freeze_config.get('freeze_strategy', 'N/A')}")
    
    def _print_yolo_specific(self, config: Dict[str, Any]):
        """Imprime información específica de YOLOv8"""
        print(f"\n🔧 Específico YOLOv8:")
        
        model_config = config.get('model', {})
        print(f"   • Variante: {model_config.get('variant', 'N/A')}")
        
        aug_config = config.get('augmentation', {})
        print(f"   • Augmentation: {aug_config.get('enabled', 'N/A')}")
        print(f"   • Mosaic: {aug_config.get('mosaic', 'N/A')}")
        print(f"   • MixUp: {aug_config.get('mixup', 'N/A')}")
        
        training_config = config.get('training', {})
        print(f"   • Patience: {training_config.get('patience', 'N/A')}")
        print(f"   • Optimizador: {training_config.get('optimizer', 'N/A')}")
    
    def compare_models(self):
        """Compara configuraciones de todos los modelos disponibles"""
        print(f"\n{'='*80}")
        print("COMPARACIÓN DE MODELOS")
        print(f"{'='*80}")
        
        print(f"{'Característica':<25} {'ResNet50':<20} {'DenseNet121':<20} {'YOLOv8':<15}")
        print(f"{'-'*80}")
        
        try:
            configs = {}
            for model_name in self.get_available_models():
                configs[model_name] = self.load_model_config(model_name)
            
            # Comparar características clave
            comparisons = [
                ('Framework', lambda c, m: self.model_info[m]['framework']),
                ('Arquitectura', lambda c, m: c['model']['architecture']),
                ('Preentrenado', lambda c, m: c['model']['pretrained']),
                ('Batch Size', lambda c, m: c['data'].get('batch_size', 'N/A')),
                ('Optimizador', lambda c, m: c['training'].get('optimizer', {}).get('type', 'N/A')),
            ]
            
            for label, extractor in comparisons:
                row = f"{label:<25}"
                for model_name in ['resnet50', 'densenet121', 'yolov8']:
                    try:
                        value = extractor(configs[model_name], model_name)
                        row += f"{str(value):<20}"
                    except:
                        row += f"{'N/A':<20}"
                print(row)
            
            print(f"{'-'*80}")
            
        except Exception as e:
            logger.error(f"Error en comparación de modelos: {e}")
    
    def export_config_summary(self, output_file: str = "config_summary.md"):
        """
        Exporta un resumen de todas las configuraciones a un archivo Markdown
        
        Args:
            output_file: Nombre del archivo de salida
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# Resumen de Configuraciones de Modelos\n\n")
                f.write("Este archivo contiene un resumen de todas las configuraciones disponibles.\n\n")
                
                for model_name in self.get_available_models():
                    config = self.load_model_config(model_name)
                    model_info = self.get_model_info(model_name)
                    
                    f.write(f"## {model_info['full_name']}\n\n")
                    f.write(f"- **Tipo**: {model_info['type']}\n")
                    f.write(f"- **Framework**: {model_info['framework']}\n")
                    f.write(f"- **Archivo de configuración**: `{self.config_files[model_name]}`\n\n")
                    
                    # Agregar detalles específicos
                    project_info = config.get('project', {})
                    f.write(f"### Configuración del Proyecto\n")
                    f.write(f"- Nombre: {project_info.get('name', 'N/A')}\n")
                    f.write(f"- Descripción: {project_info.get('description', 'N/A')}\n\n")
                    
                    # Configuración del modelo
                    model_config = config.get('model', {})
                    f.write(f"### Configuración del Modelo\n")
                    f.write(f"- Arquitectura: {model_config.get('architecture', 'N/A')}\n")
                    f.write(f"- Preentrenado: {model_config.get('pretrained', 'N/A')}\n")
                    f.write(f"- Número de clases: {model_config.get('num_classes', 'N/A')}\n\n")
                    
            logger.info(f"Resumen exportado a: {output_file}")
            
        except Exception as e:
            logger.error(f"Error al exportar resumen: {e}")
            raise

# Funciones de conveniencia para mantener compatibilidad
def load_config(config_path: str) -> Dict[str, Any]:
    """Función de conveniencia para cargar configuración"""
    loader = ConfigLoader()
    return loader.load_config(config_path)

def load_model_config(model_name: str) -> Dict[str, Any]:
    """Función de conveniencia para cargar configuración de modelo"""
    loader = ConfigLoader()
    return loader.load_model_config(model_name)

def get_available_models() -> List[str]:
    """Función de conveniencia para obtener modelos disponibles"""
    loader = ConfigLoader()
    return loader.get_available_models()

def print_config_summary(model_name: str):
    """Función de conveniencia para imprimir resumen"""
    loader = ConfigLoader()
    loader.print_config_summary(model_name)

def compare_all_models():
    """Función de conveniencia para comparar todos los modelos"""
    loader = ConfigLoader()
    loader.compare_models()

# Ejemplo de uso y testing
if __name__ == "__main__":
    # Crear instancia del cargador
    loader = ConfigLoader()
    
    print("🚀 Probando ConfigLoader con los tres modelos...")
    print(f"Modelos disponibles: {loader.get_available_models()}")
    
    # Probar cada modelo
    for model_name in loader.get_available_models():
        try:
            print(f"\n📝 Probando {model_name}...")
            config = loader.load_model_config(model_name)
            print(f"✅ {model_name} configuración cargada correctamente")
            
            # Mostrar resumen
            loader.print_config_summary(model_name)
            
        except Exception as e:
            print(f"❌ Error con {model_name}: {e}")
    
    # Comparar modelos
    print(f"\n🔍 Comparando todos los modelos...")
    loader.compare_models()
    
