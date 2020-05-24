#!groovy
pipeline {
  agent any
  stages 
    {
    stage('Build docker images') {
      steps {
        sh 'docker build -t akhilank1937/chatbuild:1.0 .'
      }
    }
    stage('Docker Hub') {
      steps 
      {
        withDockerRegistry([credentialsId: 'DockerHub', url:""])
        {
          sh 'docker push akhilank1937/chatbuild:1.0'
        }
      }
    }
  }
}