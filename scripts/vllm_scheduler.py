import docker
import time
import logging
import json
import os
import sys
from typing import Dict, Any, Optional, List
from flask import Flask, request, jsonify

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VLLMScheduler:
    """负责管理VLLM Docker容器的调度器"""
    
    def __init__(self, 
                 container_name_prefix: str = "vllm-server", 
                 default_image: str = "ghcr.io/vllm-project/vllm:latest",
                 default_port: int = 8000,
                 host_port: int = 8000):
        self.docker_client = docker.from_env()
        self.container_name_prefix = container_name_prefix
        self.default_image = default_image
        self.default_port = default_port
        self.host_port = host_port
        self.current_container_id = None
        self.container_status = "stopped"
    
    def merge_lora_weights(self,
                          model_path: str,
                          adapter_path: str,
                          export_dir: str,
                          template: str = "mistral",
                          export_size: int = 9,
                          export_device: str = "auto",
                          export_legacy_format: bool = False) -> Dict[str, Any]:
        """合并基础模型和LoRA权重"""
        try:
            # 导入tuner模块
            from llamafactory.train.tuner import export_model
            
            # 准备参数
            args = {
                "model_name_or_path": model_path,
                "adapter_name_or_path": adapter_path,
                "template": template,
                "finetuning_type": "lora",
                "export_dir": export_dir,
                "export_size": export_size,
                "export_device": export_device,
                "export_legacy_format": export_legacy_format
            }
            
            # 调用export_model合并权重
            logger.info(f"开始合并LoRA权重，基础模型: {model_path}, LoRA适配器: {adapter_path}")
            export_model(args)
            
            logger.info(f"LoRA权重合并完成，输出目录: {export_dir}")
            return {
                "success": True,
                "export_dir": export_dir,
                "message": "LoRA权重合并成功"
            }
            
        except Exception as e:
            logger.error(f"合并LoRA权重时出错: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "LoRA权重合并失败"
            }
    
    def list_vllm_containers(self) -> List[Dict[str, Any]]:
        """列出所有vllm相关容器"""
        containers = self.docker_client.containers.list(all=True)
        vllm_containers = []
        
        for container in containers:
            if self.container_name_prefix in container.name:
                vllm_containers.append({
                    "id": container.id,
                    "name": container.name,
                    "status": container.status,
                    "image": container.image.tags[0] if container.image.tags else "unknown",
                })
        
        return vllm_containers
    
    def stop_all_vllm_containers(self) -> Dict[str, Any]:
        """停止所有运行中的vllm容器"""
        containers = self.docker_client.containers.list(
            filters={"name": self.container_name_prefix}
        )
        
        stopped_count = 0
        for container in containers:
            try:
                logger.info(f"正在停止容器: {container.name}")
                container.stop(timeout=30)
                container.remove()
                stopped_count += 1
            except Exception as e:
                logger.error(f"停止容器 {container.name} 时出错: {str(e)}")
        
        self.container_status = "stopped"
        return {"success": True, "stopped_count": stopped_count}
    
    def start_vllm_container(self, 
                           model_name: str,
                           image: Optional[str] = None,
                           gpu_ids: str = "all",
                           max_model_len: int = 8192,
                           extra_args: Optional[str] = None,
                           merge_lora: bool = False,
                           base_model_path: Optional[str] = None,
                           adapter_path: Optional[str] = None,
                           export_dir: Optional[str] = None,
                           template: str = "mistral",
                           export_size: int = 9,
                           export_device: str = "auto",
                           export_legacy_format: bool = False) -> Dict[str, Any]:
        """启动新的vllm容器，可选择先合并LoRA权重"""
        try:
            # 如果需要合并LoRA权重
            if merge_lora:
                if not (base_model_path and adapter_path and export_dir):
                    return {
                        "success": False,
                        "error": "合并LoRA权重需要提供base_model_path、adapter_path和export_dir参数"
                    }
                    
                # 合并LoRA权重
                merge_result = self.merge_lora_weights(
                    model_path=base_model_path,
                    adapter_path=adapter_path,
                    export_dir=export_dir,
                    template=template,
                    export_size=export_size,
                    export_device=export_device,
                    export_legacy_format=export_legacy_format
                )
                
                if not merge_result["success"]:
                    return merge_result
                    
                # 使用合并后的模型
                model_name = export_dir
            
            # 先停止所有现有容器
            self.stop_all_vllm_containers()
            
            # 设置容器配置
            image = image or self.default_image
            container_name = f"{self.container_name_prefix}-{int(time.time())}"
            
            # 构建命令
            cmd = f"--model {model_name} --gpu-memory-utilization 0.9 --port {self.default_port}"
            
            if gpu_ids != "all":
                cmd += f" --tensor-parallel-size {len(gpu_ids.split(','))}"
            
            if max_model_len:
                cmd += f" --max-model-len {max_model_len}"
                
            if extra_args:
                cmd += f" {extra_args}"
            
            # 设置环境变量和设备
            environment = {"CUDA_VISIBLE_DEVICES": gpu_ids if gpu_ids != "all" else None}
            environment = {k: v for k, v in environment.items() if v is not None}
            
            device_requests = []
            if gpu_ids == "all":
                device_requests.append(docker.types.DeviceRequest(count=-1, capabilities=[['gpu']]))
            else:
                for gpu_id in gpu_ids.split(','):
                    device_requests.append(docker.types.DeviceRequest(
                        device_ids=[gpu_id.strip()],
                        capabilities=[['gpu']]
                    ))
            
            # 启动容器
            container = self.docker_client.containers.run(
                image=image,
                command=cmd,
                name=container_name,
                detach=True,
                ports={f"{self.default_port}/tcp": self.host_port},
                environment=environment,
                device_requests=device_requests,
                restart_policy={"Name": "unless-stopped"}
            )
            
            self.current_container_id = container.id
            self.container_status = "starting"
            
            logger.info(f"启动VLLM容器: {container_name}, ID: {container.id}")
            
            # 启动状态监控线程
            import threading
            threading.Thread(target=self._monitor_container_health, 
                            args=(self.current_container_id,)).start()
            
            return {
                "success": True, 
                "container_id": container.id,
                "container_name": container_name,
                "status": "starting",
                "api_url": f"http://localhost:{self.host_port}/v1"
            }
            
        except Exception as e:
            logger.error(f"启动VLLM容器时出错: {str(e)}")
            self.container_status = "error"
            return {
                "success": False,
                "error": str(e),
                "status": "error"
            }
    
    def _monitor_container_health(self, container_id: str):
        """监控容器健康状态的内部方法"""
        import requests
        
        max_retries = 30
        retry_delay = 10
        health_url = f"http://localhost:{self.host_port}/v1/models"
        
        for attempt in range(max_retries):
            try:
                # 检查容器是否仍在运行
                try:
                    container = self.docker_client.containers.get(container_id)
                    if container.status != "running":
                        logger.error(f"容器已停止运行: {container.status}")
                        self.container_status = "stopped"
                        return
                except Exception:
                    logger.error("无法获取容器状态，可能已被删除")
                    self.container_status = "stopped"
                    return
                
                # 检查API是否响应
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"VLLM服务已就绪! 尝试次数: {attempt+1}")
                    self.container_status = "running"
                    return
            except requests.RequestException:
                logger.info(f"VLLM服务未就绪，等待中... 尝试次数: {attempt+1}/{max_retries}")
            
            time.sleep(retry_delay)
        
        # 如果达到最大重试次数仍未就绪
        logger.error("VLLM服务未能在预期时间内就绪")
        self.container_status = "unhealthy"
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前容器状态"""
        if not self.current_container_id:
            return {"status": "stopped", "container_id": None}
        
        try:
            container = self.docker_client.containers.get(self.current_container_id)
            return {
                "status": self.container_status,
                "container_status": container.status,
                "container_id": container.id,
                "container_name": container.name,
                "image": container.image.tags[0] if container.image.tags else "unknown",
                "api_url": f"http://localhost:{self.host_port}/v1"
            }
        except docker.errors.NotFound:
            self.current_container_id = None
            self.container_status = "stopped"
            return {"status": "stopped", "container_id": None}
        except Exception as e:
            logger.error(f"获取容器状态时出错: {str(e)}")
            return {"status": "error", "error": str(e)}

# 创建Flask应用
app = Flask(__name__)
scheduler = VLLMScheduler()

@app.route("/status", methods=["GET"])
def get_status():
    """获取当前vllm容器状态"""
    return jsonify(scheduler.get_status())

@app.route("/containers", methods=["GET"])
def list_containers():
    """列出所有vllm容器"""
    return jsonify(scheduler.list_vllm_containers())

@app.route("/stop", methods=["POST"])
def stop_containers():
    """停止所有vllm容器"""
    result = scheduler.stop_all_vllm_containers()
    return jsonify(result)

@app.route("/merge_lora", methods=["POST"])
def merge_lora():
    """合并基础模型和LoRA权重"""
    data = request.json or {}
    
    if "model_path" not in data or "adapter_path" not in data or "export_dir" not in data:
        return jsonify({
            "success": False, 
            "error": "缺少必要参数: model_path, adapter_path, export_dir"
        }), 400
    
    result = scheduler.merge_lora_weights(
        model_path=data["model_path"],
        adapter_path=data["adapter_path"],
        export_dir=data["export_dir"],
        template=data.get("template", "mistral"),
        export_size=data.get("export_size", 9),
        export_device=data.get("export_device", "auto"),
        export_legacy_format=data.get("export_legacy_format", False)
    )
    
    return jsonify(result)

@app.route("/start", methods=["POST"])
def start_container():
    """启动新的vllm容器"""
    data = request.json or {}
    
    if "model_name" not in data:
        return jsonify({"success": False, "error": "缺少必要参数: model_name"}), 400
    
    # 检查是否需要合并LoRA权重
    merge_lora = data.get("merge_lora", False)
    if merge_lora and (
        "base_model_path" not in data or 
        "adapter_path" not in data or 
        "export_dir" not in data
    ):
        return jsonify({
            "success": False, 
            "error": "合并LoRA权重需要提供base_model_path、adapter_path和export_dir参数"
        }), 400
    
    result = scheduler.start_vllm_container(
        model_name=data["model_name"],
        image=data.get("image"),
        gpu_ids=data.get("gpu_ids", "all"),
        max_model_len=data.get("max_model_len", 8192),
        extra_args=data.get("extra_args"),
        merge_lora=merge_lora,
        base_model_path=data.get("base_model_path"),
        adapter_path=data.get("adapter_path"),
        export_dir=data.get("export_dir"),
        template=data.get("template", "mistral"),
        export_size=data.get("export_size", 9),
        export_device=data.get("export_device", "auto"),
        export_legacy_format=data.get("export_legacy_format", False)
    )
    
    return jsonify(result)

if __name__ == "__main__":
    # 从环境变量获取主机端口
    host_port = int(os.environ.get("VLLM_SCHEDULER_PORT", 5000))
    
    logger.info(f"启动VLLM调度器服务，监听端口: {host_port}")
    app.run(host="0.0.0.0", port=host_port) 