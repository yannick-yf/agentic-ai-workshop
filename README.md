# Atelier (3h) : Créez votre premier agent IA avec OpenAI et Agno

## Objectif
Apprendre à concevoir, développer et exécuter un agent IA en utilisant OpenAI SDK et Agno/Phidata.

## Format
- Présentation rapide
- Démonstrations en direct
- Activités pratiques
- Questions-réponses

---

## Plan détaillé

### 1er heure – Introduction et Bases des Agents IA (60 min)

#### 1.1 Comprendre les Agents IA (20 min)
##### __Qu'est-ce qu'un Agent IA ? (Définition et concepts clés)__

Un **Agent IA** est une entité qui perçoit son environnement, raisonne et agit pour atteindre des objectifs. Ils sont conçus pour être capables de penser, d'agir et d'utiliser des fonctionnalités externes et d'accéder à des données en temps réel de manière automatisée et autonome.

La relation entre les Agents IA et les **grands modèles de langage (LLMs)** est symbiotique, le LLM agissant souvent comme le cerveau de l'agent. Les Agents IA augmentent les capacités des LLMs en leur permettant d'interagir avec le monde extérieur et de dépasser les limites de leurs données d'entraînement. Par exemple, un LLM seul ne peut pas accéder à des informations en temps réel sur l'état d'une commande, mais un Agent IA peut utiliser des outils pour le faire.

##### _Composants fondamentaux d'un Agent IA_

Selon les sources, les composants clés d'un Agent IA incluent :

- **Agent Core** : Le moteur de décision, souvent basé sur un LLM, qui interprète les entrées et décide des actions. Il utilise des prompts, comme le *ReAct Prompt*, pour déterminer ses pensées et ses actions.
- **Mémoire** : La capacité de l'agent à conserver et à utiliser des informations des interactions passées. Ceci est essentiel pour le suivi des conversations et la construction de workflows.
- **Outils (Actions)** : Des fonctions externes ou des API que l'agent peut invoquer pour interagir avec le monde réel ou accéder à des données. Exemples : recherche web, accès à des bases de données, calculatrices.
- **Module de Planification (si applicable)** : Pour les agents plus complexes, la capacité à décomposer des tâches en étapes et à les exécuter séquentiellement.
- **Système de Prompt** : Un ensemble d'instructions fondamentales qui définissent le comportement, le ton et l'approche décisionnelle de l'agent.

##### Pourquoi utiliser des agents IA ?

Les Agents IA offrent de nombreux avantages et peuvent être utilisés dans divers contextes :

- **Automatisation Avancée des Tâches** : Automatisation de flux de travail complexes, allant au-delà des simples scripts.
- **Amélioration de la Productivité** : Délégation de tâches répétitives ou nécessitant de la recherche pour libérer du temps.
- **Accès et Traitement de Données en Temps Réel** : Interrogation de données à jour pour prise de décision informée.
- **Support Client Intelligent** : Réponse aux questions fréquentes, suivi des commandes, consultation de bases de connaissances.
- **Automatisation de Flux de Travail Métier** : Analyse de documents, génération de rapports, initiation d'actions basées sur des événements.
- **Analyse de Données et Recherche** : Interrogation de bases de données, recherches web, synthèse d'informations.
- **Création d'Assistants Personnels Avancés** : Planification de rendez-vous, gestion d'informations personnelles.
- **Systèmes Multi-Agents** : Collaboration entre agents spécialisés pour résoudre des problèmes complexes.

#### 1.2 Introduction aux Outils : OpenAI SDK et Agno (20 min)

##### Qu'est-ce qu'OpenAI SDK ?

L'**OpenAI SDK** est une bibliothèque Python permettant aux développeurs d'interagir avec les API d'OpenAI.

Avec l'OpenAI SDK, vous pouvez :

- Configurer votre clé d'API pour authentifier vos requêtes.
- Utiliser `openai.chat.completions.create()` pour envoyer des prompts aux modèles et obtenir des réponses.
- Définir des *prompts système* pour orienter le comportement des modèles.
- Passer des *prompts utilisateur* pour poser des questions ou donner des instructions.
- Analyser les réponses de l'API pour extraire le contenu généré.
- Utiliser des outils définis selon le format d'OpenAI pour exécuter des fonctions externes.

##### Qu'est-ce qu'Agno ?

**Agno** (anciennement *Phidata*) est un framework Python open-source conçu pour construire des Agents IA multimodaux avec mémoire, connaissances et outils. Il est simple, rapide et indépendant du modèle (*model-agnostic*).

Agno permet de créer des agents qui peuvent travailler avec du texte, des images, de l'audio et de la vidéo. Il facilite la construction d'équipes d'agents spécialisés (*multi-agents*) et offre des fonctionnalités pour la gestion de la mémoire, l'utilisation de bases de connaissances (*vector databases* pour la RAG), et la production de sorties structurées.

##### Pourquoi utiliser Agno ?

- **Simplicité et Rapidité** : Création rapide d'agents avec seulement trois lignes de code.
- **Indépendance du Modèle** : Compatible avec plusieurs fournisseurs de modèles (ex. OpenAI, Mistral, Anthropic).
- **Capacités Multimodales** : Support natif pour texte, images, audio et vidéo.
- **Support Multi-Agents** : Facilite la collaboration entre agents spécialisés.
- **Fonctionnalités Avancées** : Mémoire intégrée, gestion des connaissances (*RAG*), sorties structurées.
- **Approche IA comme Ingénierie Logicielle** : Utilisation de constructions Python standards (*if, else, while, for*).
- **Facilité d'Intégration avec des Outils** : Recherche web, finance, bases de données.
- **Interface Utilisateur (UI)** : Agno propose une UI graphique pour interagir avec les agents.

En résumé, **Agno** est un framework puissant et flexible pour développer des Agents IA intelligents et adaptatifs.

#### 1.3 Définition de votre Agent IA (20 min)
- **Atelier interactif** : Définir un cas d’usage simple
  - Objectif de l’agent
  - Entrées et sorties attendues
  - Actions/outils nécessaires
- Exemples : chatbot de support, analyseur de texte, recherche web


---

### 2eme heure – Développement Pratique d’un Agent IA (60 min)

#### 2.1 Configuration de l’environnement (15 min)
- Installation des dépendances :
  ```bash
  pip install openai agno phidata
  ```
- Gestion des clés API avec variables d’environnement
- Exécution d’un premier test simple avec OpenAI SDK

#### 2.2 Interaction avec OpenAI en Python (20 min)
- Création d’un premier agent conversationnel simple
- Structuration des prompts et rôle du system message
- Exemples de modifications des paramètres (*temperature*, *max_tokens*, etc.)

#### 2.3 Ajout d’outils à l’Agent avec Agno (25 min)
- Définition et enregistrement d’actions avec Agno
- Intégration d’une action simple (exemple : récupération de l’heure, recherche web)
- Exécution et test de l’agent avec plusieurs entrées

---

### 3eme heure – Amélioration, Scalabilité et Déploiement (60 min)

#### 3.1 Optimisation et Cas d’Utilisation Avancés (20 min)
- Ajout de mémoire pour la gestion du contexte
- Utilisation d’outils plus complexes (connexion à une base de données, génération de texte avancée)
- Discussion sur les limitations et meilleures pratiques

#### 3.2 Projet Pratique : Création de votre Agent IA (30 min)
- Développement d’un agent en binôme ou individuel
- Exemples de tâches :
  - Un chatbot assistant
  - Un analyseur de texte intelligent
  - Un agent de veille automatique
- Assistance et debugging en direct

#### 3.3 Prochaines Étapes et Ressources (10 min)
- Exploration des sujets avancés :
  - Agents multi-outils
  - RAG (Retrieval Augmented Generation)
  - Déploiement dans un environnement cloud
- Ressources pour approfondir : Documentation OpenAI et Agno, cours en ligne, GitHub de projets