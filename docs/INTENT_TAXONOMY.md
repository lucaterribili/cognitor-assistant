# Intenti base

685 intenti, 12499 esempi. Sorgente di verita': `knowledge/intents/*.yaml`.

Gli intenti sono organizzati per file in base al suffisso e splittati per entita' dominante. Rigenera con `python scripts/validate_intents.py --doc`.

| Intento | File | Esempi | Entita |
|---|---|---|---|
| `ask_accessibility` | `accessibility.yaml` | 17 |  |
| `ask_facility_accessibility` | `accessibility.yaml` | 17 | FACILITY, ZONE |
| `ask_health_service_accessibility` | `accessibility.yaml` | 17 | HEALTH_SERVICE, ORGANIZATION |
| `ask_organization_accessibility` | `accessibility.yaml` | 18 | ORGANIZATION |
| `ask_department_accessibility` | `accessibility.yaml` | 17 | DEPARTMENT |
| `ask_misc_accessibility` | `accessibility.yaml` | 17 | MISC |
| `ask_zone_accessibility` | `accessibility.yaml` | 17 | ZONE |
| `ask_activities` | `activities.yaml` | 42 |  |
| `ask_department_activities` | `activities.yaml` | 22 | DEPARTMENT |
| `ask_organization_activities` | `activities.yaml` | 18 | ORGANIZATION, ZONE |
| `ask_category_activities` | `activities.yaml` | 17 | CATEGORY |
| `ask_event_activities` | `activities.yaml` | 18 | EVENT |
| `ask_health_service_activities` | `activities.yaml` | 16 | HEALTH_SERVICE |
| `ask_document_activities` | `activities.yaml` | 17 | DOCUMENT |
| `ask_facility_activities` | `activities.yaml` | 21 | FACILITY |
| `ask_procedure_activities` | `activities.yaml` | 17 | PROCEDURE |
| `ask_address` | `address.yaml` | 18 |  |
| `ask_facility_address` | `address.yaml` | 25 | FACILITY |
| `ask_department_address` | `address.yaml` | 26 | DEPARTMENT, ORGANIZATION, ZONE |
| `ask_organization_address` | `address.yaml` | 17 | ORGANIZATION |
| `ask_address_address` | `address.yaml` | 16 | ADDRESS |
| `ask_document_address` | `address.yaml` | 16 | DOCUMENT |
| `ask_health_service_address` | `address.yaml` | 16 | HEALTH_SERVICE, ZONE |
| `ask_medical_procedure_address` | `address.yaml` | 16 | MEDICAL_PROCEDURE |
| `ask_advantages` | `advantages.yaml` | 18 |  |
| `ask_health_service_advantages` | `advantages.yaml` | 18 | HEALTH_SERVICE |
| `ask_technology_advantages` | `advantages.yaml` | 18 | TECHNOLOGY, TECH_CONCEPT |
| `ask_tech_concept_advantages` | `advantages.yaml` | 16 | TECH_CONCEPT, TECHNOLOGY |
| `ask_misc_advantages` | `advantages.yaml` | 16 | MISC, DOCUMENT |
| `ask_procedure_advantages` | `advantages.yaml` | 16 | PROCEDURE |
| `ask_department_advantages` | `advantages.yaml` | 18 | DEPARTMENT |
| `ask_document_advantages` | `advantages.yaml` | 16 | DOCUMENT |
| `ask_facility_advantages` | `advantages.yaml` | 16 | FACILITY |
| `ask_product_advantages` | `advantages.yaml` | 18 | PRODUCT |
| `ask_tax_advantages` | `advantages.yaml` | 18 | TAX |
| `affirm_confirmation` | `affirm.yaml` | 30 | DOCUMENT, FACILITY, ZONE, MEDICAL_PROCEDURE |
| `affirm_negation` | `affirm.yaml` | 25 | DOCUMENT, TECHNOLOGY, FACILITY, ZONE |
| `affirm_greeting` | `affirm.yaml` | 22 |  |
| `affirm_goodbye` | `affirm.yaml` | 20 |  |
| `affirm_gratitude` | `affirm.yaml` | 25 |  |
| `affirm_insult` | `affirm.yaml` | 25 |  |
| `ask_alternatives` | `alternatives.yaml` | 18 |  |
| `ask_technology_alternatives` | `alternatives.yaml` | 17 | TECHNOLOGY |
| `ask_document_alternatives` | `alternatives.yaml` | 16 | DOCUMENT, MISC |
| `ask_health_service_alternatives` | `alternatives.yaml` | 16 | HEALTH_SERVICE, DEPARTMENT |
| `ask_facility_alternatives` | `alternatives.yaml` | 16 | FACILITY, ZONE, HEALTH_SERVICE |
| `ask_medical_procedure_alternatives` | `alternatives.yaml` | 16 | MEDICAL_PROCEDURE |
| `ask_department_alternatives` | `alternatives.yaml` | 18 | DEPARTMENT |
| `ask_procedure_alternatives` | `alternatives.yaml` | 17 | PROCEDURE |
| `ask_tax_alternatives` | `alternatives.yaml` | 16 | TAX |
| `ask_tech_concept_alternatives` | `alternatives.yaml` | 18 | TECH_CONCEPT |
| `ask_announcements` | `announcements.yaml` | 16 |  |
| `ask_organization_announcements` | `announcements.yaml` | 17 | ORGANIZATION |
| `ask_document_announcements` | `announcements.yaml` | 16 | DOCUMENT, ROLE, ORGANIZATION |
| `ask_facility_announcements` | `announcements.yaml` | 17 | FACILITY, ZONE, DATE |
| `ask_department_announcements` | `announcements.yaml` | 18 | DEPARTMENT |
| `ask_health_service_announcements` | `announcements.yaml` | 17 | HEALTH_SERVICE |
| `ask_medical_procedure_announcements` | `announcements.yaml` | 16 | MEDICAL_PROCEDURE |
| `ask_misc_announcements` | `announcements.yaml` | 16 | MISC |
| `ask_product_announcements` | `announcements.yaml` | 17 | PRODUCT |
| `ask_appointments` | `appointments.yaml` | 17 |  |
| `ask_department_appointments` | `appointments.yaml` | 25 | DEPARTMENT, ZONE |
| `ask_role_appointments` | `appointments.yaml` | 22 | ROLE, FACILITY |
| `ask_document_appointments` | `appointments.yaml` | 16 | DOCUMENT |
| `ask_health_service_appointments` | `appointments.yaml` | 18 | HEALTH_SERVICE |
| `ask_medical_procedure_appointments` | `appointments.yaml` | 18 | MEDICAL_PROCEDURE |
| `ask_facility_appointments` | `appointments.yaml` | 24 | FACILITY |
| `ask_organization_appointments` | `appointments.yaml` | 17 | ORGANIZATION |
| `ask_procedure_appointments` | `appointments.yaml` | 17 | PROCEDURE |
| `ask_person_appointments` | `appointments.yaml` | 25 | PERSON |
| `ask_availability` | `availability.yaml` | 16 |  |
| `ask_medical_procedure_availability` | `availability.yaml` | 17 | MEDICAL_PROCEDURE, ZONE |
| `ask_facility_availability` | `availability.yaml` | 18 | FACILITY, ZONE, DATE |
| `ask_health_service_availability` | `availability.yaml` | 16 | HEALTH_SERVICE, DATE, FACILITY, ZONE |
| `ask_department_availability` | `availability.yaml` | 17 | DEPARTMENT |
| `ask_document_availability` | `availability.yaml` | 18 | DOCUMENT, MISC |
| `ask_misc_availability` | `availability.yaml` | 17 | MISC |
| `ask_role_availability` | `availability.yaml` | 18 | ROLE, ZONE |
| `ask_technology_availability` | `availability.yaml` | 16 | TECHNOLOGY |
| `ask_organization_availability` | `availability.yaml` | 18 | ORGANIZATION |
| `ask_zone_availability` | `availability.yaml` | 18 | ZONE |
| `ask_booking` | `booking.yaml` | 18 |  |
| `ask_medical_procedure_booking` | `booking.yaml` | 17 | MEDICAL_PROCEDURE, DOCUMENT, FACILITY |
| `ask_health_service_booking` | `booking.yaml` | 21 | HEALTH_SERVICE, ZONE |
| `ask_facility_booking` | `booking.yaml` | 24 | FACILITY, ZONE |
| `ask_organization_booking` | `booking.yaml` | 18 | ORGANIZATION |
| `ask_department_booking` | `booking.yaml` | 24 | DEPARTMENT |
| `ask_document_booking` | `booking.yaml` | 17 | DOCUMENT |
| `ask_misc_booking` | `booking.yaml` | 18 | MISC, FACILITY |
| `ask_event_booking` | `booking.yaml` | 18 | EVENT |
| `ask_procedure_booking` | `booking.yaml` | 17 | PROCEDURE |
| `ask_technology_booking` | `booking.yaml` | 16 | TECHNOLOGY |
| `ask_zone_booking` | `booking.yaml` | 16 | ZONE |
| `ask_person_booking` | `booking.yaml` | 25 | PERSON |
| `ask_cadastral_services` | `cadastral_services.yaml` | 16 |  |
| `ask_document_cadastral_services` | `cadastral_services.yaml` | 17 | DOCUMENT, ORGANIZATION |
| `ask_misc_cadastral_services` | `cadastral_services.yaml` | 16 | MISC, ORGANIZATION, TAX |
| `ask_department_cadastral_services` | `cadastral_services.yaml` | 16 | DEPARTMENT, MISC |
| `ask_procedure_cadastral_services` | `cadastral_services.yaml` | 18 | PROCEDURE |
| `ask_organization_cadastral_services` | `cadastral_services.yaml` | 17 | ORGANIZATION |
| `ask_technology_cadastral_services` | `cadastral_services.yaml` | 18 | TECHNOLOGY, ORGANIZATION |
| `ask_careers` | `careers.yaml` | 18 |  |
| `ask_organization_careers` | `careers.yaml` | 18 | ORGANIZATION |
| `ask_document_careers` | `careers.yaml` | 18 | DOCUMENT, ORGANIZATION, ROLE |
| `ask_department_careers` | `careers.yaml` | 18 | DEPARTMENT |
| `ask_role_careers` | `careers.yaml` | 17 | ROLE |
| `ask_technology_careers` | `careers.yaml` | 16 | TECHNOLOGY |
| `ask_causes` | `causes.yaml` | 17 |  |
| `ask_document_causes` | `causes.yaml` | 16 | DOCUMENT, ORGANIZATION, MISC |
| `ask_misc_causes` | `causes.yaml` | 18 | MISC, DOCUMENT |
| `ask_technology_causes` | `causes.yaml` | 16 | TECHNOLOGY |
| `ask_health_service_causes` | `causes.yaml` | 17 | HEALTH_SERVICE |
| `ask_medical_procedure_causes` | `causes.yaml` | 16 | MEDICAL_PROCEDURE, DATE |
| `ask_department_causes` | `causes.yaml` | 17 | DEPARTMENT |
| `ask_facility_causes` | `causes.yaml` | 16 | FACILITY |
| `ask_procedure_causes` | `causes.yaml` | 18 | PROCEDURE |
| `ask_changes` | `changes.yaml` | 18 |  |
| `ask_document_changes` | `changes.yaml` | 18 | DOCUMENT |
| `ask_facility_changes` | `changes.yaml` | 17 | FACILITY |
| `ask_health_service_changes` | `changes.yaml` | 17 | HEALTH_SERVICE, FACILITY |
| `ask_department_changes` | `changes.yaml` | 16 | DEPARTMENT |
| `ask_misc_changes` | `changes.yaml` | 16 | MISC, DOCUMENT |
| `ask_organization_changes` | `changes.yaml` | 16 | ORGANIZATION |
| `ask_role_changes` | `changes.yaml` | 17 | ROLE |
| `ask_tax_changes` | `changes.yaml` | 17 | TAX |
| `ask_tech_concept_changes` | `changes.yaml` | 16 | TECH_CONCEPT, TECHNOLOGY |
| `ask_technology_changes` | `changes.yaml` | 17 | TECHNOLOGY |
| `ask_clarification` | `clarification.yaml` | 16 |  |
| `ask_document_clarification` | `clarification.yaml` | 16 | DOCUMENT |
| `ask_technology_clarification` | `clarification.yaml` | 16 | TECHNOLOGY |
| `ask_department_clarification` | `clarification.yaml` | 17 | DEPARTMENT |
| `ask_facility_clarification` | `clarification.yaml` | 17 | FACILITY |
| `ask_health_service_clarification` | `clarification.yaml` | 16 | HEALTH_SERVICE |
| `ask_medical_procedure_clarification` | `clarification.yaml` | 18 | MEDICAL_PROCEDURE |
| `ask_misc_clarification` | `clarification.yaml` | 18 | MISC, DOCUMENT |
| `ask_organization_clarification` | `clarification.yaml` | 18 | ORGANIZATION |
| `ask_role_clarification` | `clarification.yaml` | 16 | ROLE |
| `ask_tax_clarification` | `clarification.yaml` | 18 | TAX |
| `ask_zone_clarification` | `clarification.yaml` | 16 | ZONE |
| `ask_comparison` | `comparison.yaml` | 16 |  |
| `ask_technology_comparison` | `comparison.yaml` | 18 | TECHNOLOGY |
| `ask_document_comparison` | `comparison.yaml` | 16 | DOCUMENT |
| `ask_facility_comparison` | `comparison.yaml` | 18 | FACILITY, ZONE, MEDICAL_PROCEDURE |
| `ask_health_service_comparison` | `comparison.yaml` | 16 | HEALTH_SERVICE, DEPARTMENT, MEDICAL_PROCEDURE |
| `ask_department_comparison` | `comparison.yaml` | 18 | DEPARTMENT |
| `ask_misc_comparison` | `comparison.yaml` | 18 | MISC |
| `ask_dish_comparison` | `comparison.yaml` | 17 | DISH |
| `ask_role_comparison` | `comparison.yaml` | 16 | ROLE |
| `ask_tax_comparison` | `comparison.yaml` | 16 | TAX |
| `ask_tech_concept_comparison` | `comparison.yaml` | 17 | TECH_CONCEPT |
| `ask_complaints` | `complaints.yaml` | 16 |  |
| `ask_document_complaints` | `complaints.yaml` | 18 | DOCUMENT, ORGANIZATION |
| `ask_department_complaints` | `complaints.yaml` | 16 | DEPARTMENT |
| `ask_health_service_complaints` | `complaints.yaml` | 18 | HEALTH_SERVICE |
| `ask_facility_complaints` | `complaints.yaml` | 17 | FACILITY |
| `ask_medical_procedure_complaints` | `complaints.yaml` | 16 | MEDICAL_PROCEDURE |
| `ask_misc_complaints` | `complaints.yaml` | 18 | MISC |
| `ask_organization_complaints` | `complaints.yaml` | 16 | ORGANIZATION |
| `ask_procedure_complaints` | `complaints.yaml` | 17 | PROCEDURE |
| `ask_consequences` | `consequences.yaml` | 16 |  |
| `ask_document_consequences` | `consequences.yaml` | 18 | DOCUMENT |
| `ask_procedure_consequences` | `consequences.yaml` | 17 | PROCEDURE, DOCUMENT |
| `ask_technology_consequences` | `consequences.yaml` | 17 | TECHNOLOGY |
| `ask_department_consequences` | `consequences.yaml` | 18 | DEPARTMENT |
| `ask_health_service_consequences` | `consequences.yaml` | 17 | HEALTH_SERVICE |
| `ask_medical_procedure_consequences` | `consequences.yaml` | 16 | MEDICAL_PROCEDURE |
| `ask_tech_concept_consequences` | `consequences.yaml` | 17 | TECH_CONCEPT |
| `ask_contact_human` | `contact_human.yaml` | 18 |  |
| `ask_department_contact_human` | `contact_human.yaml` | 23 | DEPARTMENT |
| `ask_organization_contact_human` | `contact_human.yaml` | 18 | ORGANIZATION |
| `ask_role_contact_human` | `contact_human.yaml` | 18 | ROLE, FACILITY |
| `ask_health_service_contact_human` | `contact_human.yaml` | 18 | HEALTH_SERVICE |
| `ask_facility_contact_human` | `contact_human.yaml` | 22 | FACILITY |
| `ask_procedure_contact_human` | `contact_human.yaml` | 16 | PROCEDURE |
| `ask_person_contact_human` | `contact_human.yaml` | 25 | PERSON |
| `ask_contacts` | `contacts.yaml` | 16 |  |
| `ask_department_contacts` | `contacts.yaml` | 25 | DEPARTMENT, ZONE |
| `ask_organization_contacts` | `contacts.yaml` | 17 | ORGANIZATION |
| `ask_facility_contacts` | `contacts.yaml` | 24 | FACILITY |
| `ask_health_service_contacts` | `contacts.yaml` | 18 | HEALTH_SERVICE |
| `ask_document_contacts` | `contacts.yaml` | 17 | DOCUMENT |
| `ask_role_contacts` | `contacts.yaml` | 24 | ROLE |
| `ask_procedure_contacts` | `contacts.yaml` | 17 | PROCEDURE |
| `ask_person_contacts` | `contacts.yaml` | 25 | PERSON |
| `ask_costs` | `costs.yaml` | 17 |  |
| `ask_document_costs` | `costs.yaml` | 16 | DOCUMENT, ORGANIZATION |
| `ask_health_service_costs` | `costs.yaml` | 18 | HEALTH_SERVICE |
| `ask_medical_procedure_costs` | `costs.yaml` | 18 | MEDICAL_PROCEDURE, DOCUMENT |
| `ask_facility_costs` | `costs.yaml` | 17 | FACILITY |
| `ask_technology_costs` | `costs.yaml` | 17 | TECHNOLOGY |
| `ask_tax_costs` | `costs.yaml` | 17 | TAX |
| `ask_transport_costs` | `costs.yaml` | 18 | TRANSPORT |
| `ask_deadlines` | `deadlines.yaml` | 16 |  |
| `ask_document_deadlines` | `deadlines.yaml` | 20 | DOCUMENT, ROLE |
| `ask_tax_deadlines` | `deadlines.yaml` | 17 | TAX, DATE |
| `ask_procedure_deadlines` | `deadlines.yaml` | 17 | PROCEDURE |
| `ask_department_deadlines` | `deadlines.yaml` | 18 | DEPARTMENT |
| `ask_health_service_deadlines` | `deadlines.yaml` | 17 | HEALTH_SERVICE |
| `ask_misc_deadlines` | `deadlines.yaml` | 18 | MISC |
| `ask_role_deadlines` | `deadlines.yaml` | 18 | ROLE |
| `ask_definition` | `definition.yaml` | 21 |  |
| `ask_document_definition` | `definition.yaml` | 15 | DOCUMENT |
| `ask_misc_definition` | `definition.yaml` | 18 | MISC |
| `ask_tax_definition` | `definition.yaml` | 16 | TAX |
| `ask_health_service_definition` | `definition.yaml` | 22 | HEALTH_SERVICE |
| `ask_tech_concept_definition` | `definition.yaml` | 18 | TECH_CONCEPT, TECHNOLOGY |
| `ask_department_definition` | `definition.yaml` | 25 | DEPARTMENT |
| `ask_medical_procedure_definition` | `definition.yaml` | 16 | MEDICAL_PROCEDURE |
| `ask_technology_definition` | `definition.yaml` | 16 | TECHNOLOGY |
| `ask_facility_definition` | `definition.yaml` | 24 | FACILITY |
| `ask_procedure_definition` | `definition.yaml` | 16 | PROCEDURE |
| `ask_organization_definition` | `definition.yaml` | 18 | ORGANIZATION |
| `ask_role_definition` | `definition.yaml` | 29 | ROLE |
| `ask_delivery` | `delivery.yaml` | 18 |  |
| `ask_document_delivery` | `delivery.yaml` | 16 | DOCUMENT |
| `ask_product_delivery` | `delivery.yaml` | 18 | PRODUCT |
| `ask_health_service_delivery` | `delivery.yaml` | 18 | HEALTH_SERVICE |
| `ask_misc_delivery` | `delivery.yaml` | 18 | MISC |
| `ask_organization_delivery` | `delivery.yaml` | 18 | ORGANIZATION, TECHNOLOGY |
| `ask_description` | `description.yaml` | 38 |  |
| `ask_facility_description` | `description.yaml` | 18 | FACILITY |
| `ask_health_service_description` | `description.yaml` | 16 | HEALTH_SERVICE |
| `ask_department_description` | `description.yaml` | 18 | DEPARTMENT |
| `ask_technology_description` | `description.yaml` | 17 | TECHNOLOGY |
| `ask_document_description` | `description.yaml` | 18 | DOCUMENT |
| `ask_misc_description` | `description.yaml` | 17 | MISC |
| `ask_procedure_description` | `description.yaml` | 18 | PROCEDURE |
| `ask_event_description` | `description.yaml` | 17 | EVENT |
| `ask_tech_concept_description` | `description.yaml` | 16 | TECH_CONCEPT |
| `ask_directions` | `directions.yaml` | 17 |  |
| `ask_facility_directions` | `directions.yaml` | 18 | FACILITY, TRANSPORT, ZONE |
| `ask_department_directions` | `directions.yaml` | 17 | DEPARTMENT, TRANSPORT |
| `ask_organization_directions` | `directions.yaml` | 18 | ORGANIZATION, LOCATION, TRANSPORT |
| `ask_zone_directions` | `directions.yaml` | 17 | ZONE, TRANSPORT |
| `ask_location_directions` | `directions.yaml` | 16 | LOCATION, ZONE |
| `ask_disadvantages` | `disadvantages.yaml` | 18 |  |
| `ask_technology_disadvantages` | `disadvantages.yaml` | 16 | TECHNOLOGY |
| `ask_health_service_disadvantages` | `disadvantages.yaml` | 16 | HEALTH_SERVICE |
| `ask_tech_concept_disadvantages` | `disadvantages.yaml` | 18 | TECH_CONCEPT |
| `ask_document_disadvantages` | `disadvantages.yaml` | 18 | DOCUMENT |
| `ask_misc_disadvantages` | `disadvantages.yaml` | 17 | MISC |
| `ask_facility_disadvantages` | `disadvantages.yaml` | 16 | FACILITY |
| `ask_procedure_disadvantages` | `disadvantages.yaml` | 18 | PROCEDURE |
| `ask_tax_disadvantages` | `disadvantages.yaml` | 16 | TAX |
| `ask_discounts` | `discounts.yaml` | 17 |  |
| `ask_document_discounts` | `discounts.yaml` | 17 | DOCUMENT |
| `ask_misc_discounts` | `discounts.yaml` | 18 | MISC |
| `ask_medical_procedure_discounts` | `discounts.yaml` | 18 | MEDICAL_PROCEDURE |
| `ask_organization_discounts` | `discounts.yaml` | 17 | ORGANIZATION |
| `ask_procedure_discounts` | `discounts.yaml` | 17 | PROCEDURE |
| `ask_health_service_discounts` | `discounts.yaml` | 18 | HEALTH_SERVICE |
| `ask_product_discounts` | `discounts.yaml` | 16 | PRODUCT |
| `ask_tax_discounts` | `discounts.yaml` | 18 | TAX |
| `ask_technology_discounts` | `discounts.yaml` | 18 | TECHNOLOGY |
| `ask_documentation` | `documentation.yaml` | 36 |  |
| `ask_document_documentation` | `documentation.yaml` | 29 | DOCUMENT, ORGANIZATION |
| `ask_health_service_documentation` | `documentation.yaml` | 16 | HEALTH_SERVICE |
| `ask_misc_documentation` | `documentation.yaml` | 17 | MISC |
| `ask_procedure_documentation` | `documentation.yaml` | 16 | PROCEDURE |
| `ask_medical_procedure_documentation` | `documentation.yaml` | 18 | MEDICAL_PROCEDURE, DOCUMENT |
| `ask_downloads` | `downloads.yaml` | 39 | DATE |
| `ask_document_downloads` | `downloads.yaml` | 17 | DOCUMENT, MISC |
| `ask_technology_downloads` | `downloads.yaml` | 16 | TECHNOLOGY |
| `ask_department_downloads` | `downloads.yaml` | 16 | DEPARTMENT |
| `ask_procedure_downloads` | `downloads.yaml` | 18 | PROCEDURE |
| `ask_duration` | `duration.yaml` | 18 |  |
| `ask_medical_procedure_duration` | `duration.yaml` | 16 | MEDICAL_PROCEDURE, DOCUMENT |
| `ask_document_duration` | `duration.yaml` | 17 | DOCUMENT |
| `ask_health_service_duration` | `duration.yaml` | 18 | HEALTH_SERVICE |
| `ask_technology_duration` | `duration.yaml` | 17 | TECHNOLOGY |
| `ask_facility_duration` | `duration.yaml` | 18 | FACILITY |
| `ask_misc_duration` | `duration.yaml` | 16 | MISC |
| `ask_organization_duration` | `duration.yaml` | 18 | ORGANIZATION |
| `ask_procedure_duration` | `duration.yaml` | 18 | PROCEDURE, ORGANIZATION |
| `ask_education` | `education.yaml` | 16 |  |
| `ask_technology_education` | `education.yaml` | 16 | TECHNOLOGY |
| `ask_health_service_education` | `education.yaml` | 17 | HEALTH_SERVICE |
| `ask_language_education` | `education.yaml` | 18 | LANGUAGE, ORGANIZATION |
| `ask_misc_education` | `education.yaml` | 18 | MISC, ROLE |
| `ask_tech_concept_education` | `education.yaml` | 17 | TECH_CONCEPT |
| `ask_category_education` | `education.yaml` | 16 | CATEGORY |
| `ask_event_education` | `education.yaml` | 17 | EVENT, ORGANIZATION |
| `ask_organization_education` | `education.yaml` | 16 | ORGANIZATION |
| `ask_eligibility` | `eligibility.yaml` | 39 |  |
| `ask_document_eligibility` | `eligibility.yaml` | 16 | DOCUMENT, MEDICAL_PROCEDURE |
| `ask_health_service_eligibility` | `eligibility.yaml` | 16 | HEALTH_SERVICE |
| `ask_misc_eligibility` | `eligibility.yaml` | 17 | MISC, ZONE |
| `ask_facility_eligibility` | `eligibility.yaml` | 17 | FACILITY, ZONE |
| `ask_department_eligibility` | `eligibility.yaml` | 17 | DEPARTMENT, DOCUMENT |
| `ask_organization_eligibility` | `eligibility.yaml` | 16 | ORGANIZATION |
| `ask_procedure_eligibility` | `eligibility.yaml` | 16 | PROCEDURE |
| `ask_role_eligibility` | `eligibility.yaml` | 17 | ROLE |
| `ask_email` | `email.yaml` | 17 |  |
| `ask_department_email` | `email.yaml` | 26 | DEPARTMENT, FACILITY, DOCUMENT, ORGANIZATION |
| `ask_organization_email` | `email.yaml` | 17 | ORGANIZATION, DOCUMENT |
| `ask_document_email` | `email.yaml` | 16 | DOCUMENT |
| `ask_health_service_email` | `email.yaml` | 16 | HEALTH_SERVICE |
| `ask_facility_email` | `email.yaml` | 25 | FACILITY |
| `ask_procedure_email` | `email.yaml` | 17 | PROCEDURE, DOCUMENT |
| `ask_role_email` | `email.yaml` | 25 | ROLE |
| `ask_person_email` | `email.yaml` | 25 | PERSON |
| `ask_events` | `events.yaml` | 16 |  |
| `ask_event_events` | `events.yaml` | 18 | EVENT, FACILITY, ZONE |
| `ask_zone_events` | `events.yaml` | 17 | ZONE |
| `ask_organization_events` | `events.yaml` | 16 | ORGANIZATION, PROCEDURE |
| `ask_category_events` | `events.yaml` | 17 | CATEGORY, ZONE |
| `ask_health_service_events` | `events.yaml` | 17 | HEALTH_SERVICE, ZONE |
| `ask_document_events` | `events.yaml` | 17 | DOCUMENT |
| `ask_facility_events` | `events.yaml` | 18 | FACILITY |
| `ask_misc_events` | `events.yaml` | 18 | MISC |
| `ask_examples` | `examples.yaml` | 16 |  |
| `ask_document_examples` | `examples.yaml` | 17 | DOCUMENT, TAX |
| `ask_technology_examples` | `examples.yaml` | 16 | TECHNOLOGY |
| `ask_tech_concept_examples` | `examples.yaml` | 18 | TECH_CONCEPT, TECHNOLOGY |
| `ask_procedure_examples` | `examples.yaml` | 18 | PROCEDURE, ORGANIZATION |
| `ask_department_examples` | `examples.yaml` | 17 | DEPARTMENT |
| `ask_misc_examples` | `examples.yaml` | 18 | MISC |
| `ask_explanation` | `explanation.yaml` | 40 | DATE |
| `ask_tech_concept_explanation` | `explanation.yaml` | 16 | TECH_CONCEPT, TECHNOLOGY |
| `ask_health_service_explanation` | `explanation.yaml` | 16 | HEALTH_SERVICE |
| `ask_misc_explanation` | `explanation.yaml` | 18 | MISC, TAX |
| `ask_facility_explanation` | `explanation.yaml` | 16 | FACILITY |
| `ask_tax_explanation` | `explanation.yaml` | 17 | TAX |
| `ask_procedure_explanation` | `explanation.yaml` | 16 | PROCEDURE, ZONE |
| `ask_technology_explanation` | `explanation.yaml` | 16 | TECHNOLOGY, TECH_CONCEPT |
| `ask_department_explanation` | `explanation.yaml` | 16 | DEPARTMENT |
| `ask_document_explanation` | `explanation.yaml` | 18 | DOCUMENT |
| `ask_medical_procedure_explanation` | `explanation.yaml` | 17 | MEDICAL_PROCEDURE |
| `ask_forms` | `forms.yaml` | 34 |  |
| `ask_document_forms` | `forms.yaml` | 17 | DOCUMENT, DEPARTMENT, HEALTH_SERVICE |
| `ask_health_service_forms` | `forms.yaml` | 17 | HEALTH_SERVICE |
| `ask_procedure_forms` | `forms.yaml` | 17 | PROCEDURE, DOCUMENT |
| `ask_misc_forms` | `forms.yaml` | 17 | MISC |
| `ask_department_forms` | `forms.yaml` | 17 | DEPARTMENT |
| `ask_tax_forms` | `forms.yaml` | 16 | TAX |
| `ask_general_information` | `general_information.yaml` | 38 |  |
| `ask_department_general_information` | `general_information.yaml` | 18 | DEPARTMENT, FACILITY, ZONE |
| `ask_facility_general_information` | `general_information.yaml` | 16 | FACILITY |
| `ask_organization_general_information` | `general_information.yaml` | 17 | ORGANIZATION |
| `ask_technology_general_information` | `general_information.yaml` | 18 | TECHNOLOGY |
| `ask_health_service_general_information` | `general_information.yaml` | 17 | HEALTH_SERVICE |
| `ask_misc_general_information` | `general_information.yaml` | 17 | MISC |
| `ask_zone_general_information` | `general_information.yaml` | 16 | ZONE |
| `ask_category_general_information` | `general_information.yaml` | 17 | CATEGORY |
| `ask_document_general_information` | `general_information.yaml` | 16 | DOCUMENT |
| `ask_procedure_general_information` | `general_information.yaml` | 18 | PROCEDURE |
| `ask_tech_concept_general_information` | `general_information.yaml` | 18 | TECH_CONCEPT |
| `ask_help` | `help.yaml` | 19 |  |
| `ask_organization_help` | `help.yaml` | 18 | ORGANIZATION |
| `ask_document_help` | `help.yaml` | 18 | DOCUMENT, TAX, PROCEDURE |
| `ask_facility_help` | `help.yaml` | 16 | FACILITY |
| `ask_technology_help` | `help.yaml` | 18 | TECHNOLOGY |
| `ask_history` | `history.yaml` | 39 |  |
| `ask_facility_history` | `history.yaml` | 17 | FACILITY, ZONE |
| `ask_organization_history` | `history.yaml` | 16 | ORGANIZATION |
| `ask_technology_history` | `history.yaml` | 18 | TECHNOLOGY |
| `ask_document_history` | `history.yaml` | 17 | DOCUMENT |
| `ask_department_history` | `history.yaml` | 18 | DEPARTMENT |
| `ask_health_service_history` | `history.yaml` | 17 | HEALTH_SERVICE |
| `ask_person_history` | `history.yaml` | 18 | PERSON |
| `ask_procedure_history` | `history.yaml` | 18 | PROCEDURE, COUNTRY |
| `ask_tax_history` | `history.yaml` | 17 | TAX |
| `ask_tech_concept_history` | `history.yaml` | 17 | TECH_CONCEPT |
| `ask_zone_history` | `history.yaml` | 18 | ZONE |
| `ask_how_to` | `how_to.yaml` | 44 |  |
| `ask_document_how_to` | `how_to.yaml` | 23 | DOCUMENT, TECHNOLOGY |
| `ask_technology_how_to` | `how_to.yaml` | 17 | TECHNOLOGY |
| `ask_health_service_how_to` | `how_to.yaml` | 17 | HEALTH_SERVICE, FACILITY |
| `ask_procedure_how_to` | `how_to.yaml` | 18 | PROCEDURE, ZONE |
| `ask_misc_how_to` | `how_to.yaml` | 17 | MISC |
| `ask_role_how_to` | `how_to.yaml` | 18 | ROLE |
| `ask_department_how_to` | `how_to.yaml` | 18 | DEPARTMENT |
| `ask_facility_how_to` | `how_to.yaml` | 17 | FACILITY |
| `ask_medical_procedure_how_to` | `how_to.yaml` | 17 | MEDICAL_PROCEDURE |
| `ask_organization_how_to` | `how_to.yaml` | 16 | ORGANIZATION |
| `ask_payment_method_how_to` | `how_to.yaml` | 16 | PAYMENT_METHOD |
| `ask_product_how_to` | `how_to.yaml` | 17 | PRODUCT |
| `ask_tax_how_to` | `how_to.yaml` | 16 | TAX, PAYMENT_METHOD |
| `ask_instructions` | `instructions.yaml` | 40 |  |
| `ask_document_instructions` | `instructions.yaml` | 17 | DOCUMENT |
| `ask_medical_procedure_instructions` | `instructions.yaml` | 18 | MEDICAL_PROCEDURE |
| `ask_health_service_instructions` | `instructions.yaml` | 17 | HEALTH_SERVICE |
| `ask_misc_instructions` | `instructions.yaml` | 16 | MISC, DOCUMENT, TAX |
| `ask_technology_instructions` | `instructions.yaml` | 17 | TECHNOLOGY |
| `ask_tech_concept_instructions` | `instructions.yaml` | 16 | TECH_CONCEPT, TECHNOLOGY |
| `ask_department_instructions` | `instructions.yaml` | 17 | DEPARTMENT |
| `ask_procedure_instructions` | `instructions.yaml` | 18 | PROCEDURE |
| `ask_tax_instructions` | `instructions.yaml` | 18 | TAX, DOCUMENT |
| `ask_location` | `location.yaml` | 16 |  |
| `ask_facility_location` | `location.yaml` | 19 | FACILITY, ZONE |
| `ask_department_location` | `location.yaml` | 16 | DEPARTMENT, ZONE |
| `ask_organization_location` | `location.yaml` | 16 | ORGANIZATION, ZONE, DOCUMENT |
| `ask_zone_location` | `location.yaml` | 17 | ZONE |
| `ask_document_location` | `location.yaml` | 18 | DOCUMENT |
| `ask_health_service_location` | `location.yaml` | 16 | HEALTH_SERVICE |
| `ask_maintenance` | `maintenance.yaml` | 17 |  |
| `ask_technology_maintenance` | `maintenance.yaml` | 16 | TECHNOLOGY |
| `ask_organization_maintenance` | `maintenance.yaml` | 16 | ORGANIZATION |
| `ask_product_maintenance` | `maintenance.yaml` | 17 | PRODUCT |
| `ask_facility_maintenance` | `maintenance.yaml` | 17 | FACILITY |
| `ask_document_maintenance` | `maintenance.yaml` | 17 | DOCUMENT, ORGANIZATION |
| `ask_misc_maintenance` | `maintenance.yaml` | 16 | MISC, DOCUMENT |
| `ask_procedure_maintenance` | `maintenance.yaml` | 17 | PROCEDURE |
| `ask_tech_concept_maintenance` | `maintenance.yaml` | 17 | TECH_CONCEPT |
| `ask_mission` | `mission.yaml` | 40 |  |
| `ask_organization_mission` | `mission.yaml` | 18 | ORGANIZATION |
| `ask_department_mission` | `mission.yaml` | 16 | DEPARTMENT |
| `ask_facility_mission` | `mission.yaml` | 18 | FACILITY |
| `ask_document_mission` | `mission.yaml` | 17 | DOCUMENT |
| `ask_health_service_mission` | `mission.yaml` | 18 | HEALTH_SERVICE, MISC |
| `ask_misc_mission` | `mission.yaml` | 16 | MISC |
| `ask_technology_mission` | `mission.yaml` | 18 | TECHNOLOGY |
| `ask_news` | `news.yaml` | 17 |  |
| `ask_health_service_news` | `news.yaml` | 16 | HEALTH_SERVICE, ZONE, MISC, FACILITY |
| `ask_zone_news` | `news.yaml` | 17 | ZONE |
| `ask_document_news` | `news.yaml` | 16 | DOCUMENT |
| `ask_organization_news` | `news.yaml` | 18 | ORGANIZATION, DOCUMENT |
| `ask_technology_news` | `news.yaml` | 17 | TECHNOLOGY |
| `ask_facility_news` | `news.yaml` | 18 | FACILITY |
| `ask_tech_concept_news` | `news.yaml` | 18 | TECH_CONCEPT |
| `ask_department_news` | `news.yaml` | 17 | DEPARTMENT |
| `ask_misc_news` | `news.yaml` | 18 | MISC |
| `ask_procedure_news` | `news.yaml` | 17 | PROCEDURE |
| `ask_role_news` | `news.yaml` | 16 | ROLE |
| `ask_tax_news` | `news.yaml` | 16 | TAX |
| `ask_objectives` | `objectives.yaml` | 42 | DATE |
| `ask_document_objectives` | `objectives.yaml` | 17 | DOCUMENT |
| `ask_facility_objectives` | `objectives.yaml` | 17 | FACILITY |
| `ask_health_service_objectives` | `objectives.yaml` | 16 | HEALTH_SERVICE, ZONE |
| `ask_misc_objectives` | `objectives.yaml` | 18 | MISC |
| `ask_technology_objectives` | `objectives.yaml` | 16 | TECHNOLOGY |
| `ask_event_objectives` | `objectives.yaml` | 18 | EVENT |
| `ask_organization_objectives` | `objectives.yaml` | 18 | ORGANIZATION |
| `ask_procedure_objectives` | `objectives.yaml` | 18 | PROCEDURE |
| `ask_tech_concept_objectives` | `objectives.yaml` | 17 | TECH_CONCEPT |
| `ask_obligations` | `obligations.yaml` | 16 |  |
| `ask_document_obligations` | `obligations.yaml` | 18 | DOCUMENT, MISC |
| `ask_procedure_obligations` | `obligations.yaml` | 17 | PROCEDURE |
| `ask_department_obligations` | `obligations.yaml` | 17 | DEPARTMENT |
| `ask_medical_procedure_obligations` | `obligations.yaml` | 18 | MEDICAL_PROCEDURE |
| `ask_organization_obligations` | `obligations.yaml` | 16 | ORGANIZATION |
| `ask_product_obligations` | `obligations.yaml` | 16 | PRODUCT |
| `ask_role_obligations` | `obligations.yaml` | 17 | ROLE |
| `ask_opening_hours` | `opening_hours.yaml` | 18 | DATE |
| `ask_facility_opening_hours` | `opening_hours.yaml` | 26 | FACILITY, ZONE, DATE |
| `ask_department_opening_hours` | `opening_hours.yaml` | 27 | DEPARTMENT, DATE, TIME |
| `ask_health_service_opening_hours` | `opening_hours.yaml` | 16 | HEALTH_SERVICE, FACILITY, TIME, DATE |
| `ask_document_opening_hours` | `opening_hours.yaml` | 18 | DOCUMENT, FACILITY |
| `ask_organization_opening_hours` | `opening_hours.yaml` | 17 | ORGANIZATION |
| `ask_medical_procedure_opening_hours` | `opening_hours.yaml` | 16 | MEDICAL_PROCEDURE, FACILITY |
| `ask_organization` | `organization.yaml` | 43 |  |
| `ask_organization_organization` | `organization.yaml` | 17 | ORGANIZATION |
| `ask_department_organization` | `organization.yaml` | 23 | DEPARTMENT, ZONE, ORGANIZATION |
| `ask_facility_organization` | `organization.yaml` | 21 | FACILITY |
| `ask_zone_organization` | `organization.yaml` | 18 | ZONE |
| `ask_payments` | `payments.yaml` | 16 |  |
| `ask_document_payments` | `payments.yaml` | 17 | DOCUMENT, PAYMENT_METHOD, ORGANIZATION |
| `ask_payment_method_payments` | `payments.yaml` | 18 | PAYMENT_METHOD |
| `ask_tax_payments` | `payments.yaml` | 16 | TAX, ORGANIZATION, PAYMENT_METHOD, DOCUMENT |
| `ask_medical_procedure_payments` | `payments.yaml` | 16 | MEDICAL_PROCEDURE |
| `ask_facility_payments` | `payments.yaml` | 18 | FACILITY, PAYMENT_METHOD |
| `ask_health_service_payments` | `payments.yaml` | 17 | HEALTH_SERVICE, PAYMENT_METHOD |
| `ask_organization_payments` | `payments.yaml` | 17 | ORGANIZATION, PAYMENT_METHOD |
| `ask_product_payments` | `payments.yaml` | 18 | PRODUCT |
| `ask_phone` | `phone.yaml` | 17 |  |
| `ask_department_phone` | `phone.yaml` | 28 | DEPARTMENT |
| `ask_health_service_phone` | `phone.yaml` | 17 | HEALTH_SERVICE |
| `ask_organization_phone` | `phone.yaml` | 17 | ORGANIZATION |
| `ask_facility_phone` | `phone.yaml` | 24 | FACILITY, ZONE |
| `ask_document_phone` | `phone.yaml` | 18 | DOCUMENT |
| `ask_medical_procedure_phone` | `phone.yaml` | 16 | MEDICAL_PROCEDURE |
| `ask_misc_phone` | `phone.yaml` | 18 | MISC |
| `ask_person_phone` | `phone.yaml` | 25 | PERSON |
| `ask_role_phone` | `phone.yaml` | 20 | ROLE |
| `ask_prices` | `prices.yaml` | 16 |  |
| `ask_document_prices` | `prices.yaml` | 17 | DOCUMENT, MEDICAL_PROCEDURE, ORGANIZATION, ROLE |
| `ask_medical_procedure_prices` | `prices.yaml` | 17 | MEDICAL_PROCEDURE |
| `ask_misc_prices` | `prices.yaml` | 16 | MISC |
| `ask_tax_prices` | `prices.yaml` | 16 | TAX, DOCUMENT |
| `ask_product_prices` | `prices.yaml` | 18 | PRODUCT |
| `ask_organization_prices` | `prices.yaml` | 18 | ORGANIZATION |
| `ask_technology_prices` | `prices.yaml` | 17 | TECHNOLOGY |
| `ask_privacy` | `privacy.yaml` | 17 |  |
| `ask_document_privacy` | `privacy.yaml` | 18 | DOCUMENT |
| `ask_health_service_privacy` | `privacy.yaml` | 16 | HEALTH_SERVICE, DOCUMENT |
| `ask_organization_privacy` | `privacy.yaml` | 17 | ORGANIZATION, DOCUMENT |
| `ask_department_privacy` | `privacy.yaml` | 18 | DEPARTMENT |
| `ask_role_privacy` | `privacy.yaml` | 17 | ROLE, ORGANIZATION, MISC |
| `ask_misc_privacy` | `privacy.yaml` | 17 | MISC |
| `ask_procedures` | `procedures.yaml` | 42 | COUNTRY |
| `ask_procedure_procedures` | `procedures.yaml` | 18 | PROCEDURE, DOCUMENT |
| `ask_document_procedures` | `procedures.yaml` | 18 | DOCUMENT, ORGANIZATION |
| `ask_health_service_procedures` | `procedures.yaml` | 17 | HEALTH_SERVICE |
| `ask_medical_procedure_procedures` | `procedures.yaml` | 18 | MEDICAL_PROCEDURE, DOCUMENT |
| `ask_misc_procedures` | `procedures.yaml` | 18 | MISC |
| `ask_role_procedures` | `procedures.yaml` | 18 | ROLE |
| `ask_technology_procedures` | `procedures.yaml` | 17 | TECHNOLOGY |
| `ask_products` | `products.yaml` | 45 |  |
| `ask_product_products` | `products.yaml` | 17 | PRODUCT |
| `ask_document_products` | `products.yaml` | 17 | DOCUMENT |
| `ask_health_service_products` | `products.yaml` | 16 | HEALTH_SERVICE |
| `ask_organization_products` | `products.yaml` | 17 | ORGANIZATION |
| `ask_recommendations` | `recommendations.yaml` | 17 |  |
| `ask_document_recommendations` | `recommendations.yaml` | 16 | DOCUMENT, MISC |
| `ask_technology_recommendations` | `recommendations.yaml` | 17 | TECHNOLOGY |
| `ask_facility_recommendations` | `recommendations.yaml` | 18 | FACILITY, ZONE |
| `ask_health_service_recommendations` | `recommendations.yaml` | 16 | HEALTH_SERVICE |
| `ask_misc_recommendations` | `recommendations.yaml` | 18 | MISC |
| `ask_product_recommendations` | `recommendations.yaml` | 17 | PRODUCT |
| `ask_zone_recommendations` | `recommendations.yaml` | 18 | ZONE |
| `ask_department_recommendations` | `recommendations.yaml` | 17 | DEPARTMENT |
| `ask_procedure_recommendations` | `recommendations.yaml` | 16 | PROCEDURE |
| `ask_tax_recommendations` | `recommendations.yaml` | 16 | TAX |
| `ask_tech_concept_recommendations` | `recommendations.yaml` | 16 | TECH_CONCEPT |
| `ask_refunds` | `refunds.yaml` | 16 |  |
| `ask_document_refunds` | `refunds.yaml` | 18 | DOCUMENT |
| `ask_tax_refunds` | `refunds.yaml` | 16 | TAX |
| `ask_health_service_refunds` | `refunds.yaml` | 16 | HEALTH_SERVICE |
| `ask_medical_procedure_refunds` | `refunds.yaml` | 17 | MEDICAL_PROCEDURE |
| `ask_misc_refunds` | `refunds.yaml` | 18 | MISC, ROLE |
| `ask_registration` | `registration.yaml` | 17 |  |
| `ask_health_service_registration` | `registration.yaml` | 17 | HEALTH_SERVICE |
| `ask_document_registration` | `registration.yaml` | 16 | DOCUMENT, ORGANIZATION |
| `ask_misc_registration` | `registration.yaml` | 18 | MISC |
| `ask_facility_registration` | `registration.yaml` | 18 | FACILITY |
| `ask_organization_registration` | `registration.yaml` | 18 | ORGANIZATION |
| `ask_role_registration` | `registration.yaml` | 18 | ROLE |
| `ask_department_registration` | `registration.yaml` | 17 | DEPARTMENT |
| `ask_event_registration` | `registration.yaml` | 16 | EVENT, MISC |
| `ask_technology_registration` | `registration.yaml` | 17 | TECHNOLOGY |
| `ask_transport_registration` | `registration.yaml` | 18 | TRANSPORT |
| `ask_regulations` | `regulations.yaml` | 17 | COUNTRY |
| `ask_document_regulations` | `regulations.yaml` | 16 | DOCUMENT, TAX |
| `ask_health_service_regulations` | `regulations.yaml` | 18 | HEALTH_SERVICE |
| `ask_procedure_regulations` | `regulations.yaml` | 16 | PROCEDURE |
| `ask_misc_regulations` | `regulations.yaml` | 17 | MISC |
| `ask_organization_regulations` | `regulations.yaml` | 17 | ORGANIZATION |
| `ask_requirements` | `requirements.yaml` | 39 |  |
| `ask_document_requirements` | `requirements.yaml` | 16 | DOCUMENT, MISC, MEDICAL_PROCEDURE |
| `ask_health_service_requirements` | `requirements.yaml` | 17 | HEALTH_SERVICE |
| `ask_technology_requirements` | `requirements.yaml` | 18 | TECHNOLOGY |
| `ask_facility_requirements` | `requirements.yaml` | 17 | FACILITY |
| `ask_misc_requirements` | `requirements.yaml` | 17 | MISC |
| `ask_tax_requirements` | `requirements.yaml` | 16 | TAX |
| `ask_department_requirements` | `requirements.yaml` | 18 | DEPARTMENT |
| `ask_medical_procedure_requirements` | `requirements.yaml` | 17 | MEDICAL_PROCEDURE |
| `ask_procedure_requirements` | `requirements.yaml` | 18 | PROCEDURE, DOCUMENT |
| `ask_transport_requirements` | `requirements.yaml` | 18 | TRANSPORT |
| `ask_rights` | `rights.yaml` | 18 |  |
| `ask_document_rights` | `rights.yaml` | 18 | DOCUMENT |
| `ask_procedure_rights` | `rights.yaml` | 18 | PROCEDURE, ORGANIZATION |
| `ask_health_service_rights` | `rights.yaml` | 17 | HEALTH_SERVICE |
| `ask_organization_rights` | `rights.yaml` | 18 | ORGANIZATION |
| `ask_department_rights` | `rights.yaml` | 18 | DEPARTMENT |
| `ask_medical_procedure_rights` | `rights.yaml` | 17 | MEDICAL_PROCEDURE |
| `ask_risks` | `risks.yaml` | 17 |  |
| `ask_document_risks` | `risks.yaml` | 18 | DOCUMENT |
| `ask_medical_procedure_risks` | `risks.yaml` | 16 | MEDICAL_PROCEDURE |
| `ask_misc_risks` | `risks.yaml` | 16 | MISC |
| `ask_procedure_risks` | `risks.yaml` | 18 | PROCEDURE, MISC |
| `ask_tech_concept_risks` | `risks.yaml` | 16 | TECH_CONCEPT |
| `ask_health_service_risks` | `risks.yaml` | 16 | HEALTH_SERVICE |
| `ask_product_risks` | `risks.yaml` | 18 | PRODUCT |
| `ask_technology_risks` | `risks.yaml` | 18 | TECHNOLOGY |
| `ask_safety` | `safety.yaml` | 16 |  |
| `ask_misc_safety` | `safety.yaml` | 18 | MISC, FACILITY, ORGANIZATION |
| `ask_technology_safety` | `safety.yaml` | 16 | TECHNOLOGY, TECH_CONCEPT |
| `ask_department_safety` | `safety.yaml` | 18 | DEPARTMENT |
| `ask_document_safety` | `safety.yaml` | 18 | DOCUMENT, TECHNOLOGY, ORGANIZATION |
| `ask_facility_safety` | `safety.yaml` | 16 | FACILITY |
| `ask_health_service_safety` | `safety.yaml` | 17 | HEALTH_SERVICE |
| `ask_organization_safety` | `safety.yaml` | 17 | ORGANIZATION |
| `ask_role_safety` | `safety.yaml` | 18 | ROLE |
| `ask_tech_concept_safety` | `safety.yaml` | 17 | TECH_CONCEPT, TECHNOLOGY |
| `ask_schedule` | `schedule.yaml` | 16 | DATE |
| `ask_health_service_schedule` | `schedule.yaml` | 17 | HEALTH_SERVICE, FACILITY, ZONE |
| `ask_zone_schedule` | `schedule.yaml` | 16 | ZONE, DATE |
| `ask_event_schedule` | `schedule.yaml` | 16 | EVENT, FACILITY |
| `ask_medical_procedure_schedule` | `schedule.yaml` | 17 | MEDICAL_PROCEDURE |
| `ask_tax_schedule` | `schedule.yaml` | 17 | TAX |
| `ask_document_schedule` | `schedule.yaml` | 18 | DOCUMENT |
| `ask_procedure_schedule` | `schedule.yaml` | 18 | PROCEDURE |
| `ask_department_schedule` | `schedule.yaml` | 18 | DEPARTMENT |
| `ask_facility_schedule` | `schedule.yaml` | 16 | FACILITY |
| `ask_misc_schedule` | `schedule.yaml` | 17 | MISC |
| `ask_organization_schedule` | `schedule.yaml` | 18 | ORGANIZATION |
| `ask_services` | `services.yaml` | 42 |  |
| `ask_health_service_services` | `services.yaml` | 21 | HEALTH_SERVICE, ZONE |
| `ask_organization_services` | `services.yaml` | 18 | ORGANIZATION, TECHNOLOGY |
| `ask_facility_services` | `services.yaml` | 22 | FACILITY, ZONE |
| `ask_department_services` | `services.yaml` | 24 | DEPARTMENT |
| `ask_misc_services` | `services.yaml` | 17 | MISC |
| `ask_document_services` | `services.yaml` | 18 | DOCUMENT |
| `ask_procedure_services` | `services.yaml` | 17 | PROCEDURE |
| `ask_medical_procedure_services` | `services.yaml` | 18 | MEDICAL_PROCEDURE |
| `ask_technology_services` | `services.yaml` | 16 | TECHNOLOGY |
| `ask_staff` | `staff.yaml` | 33 |  |
| `ask_role_staff` | `staff.yaml` | 28 | ROLE, ORGANIZATION, ZONE, FACILITY |
| `ask_department_staff` | `staff.yaml` | 29 | DEPARTMENT |
| `ask_person_staff` | `staff.yaml` | 41 | PERSON |
| `ask_organization_staff` | `staff.yaml` | 16 | ORGANIZATION |
| `ask_document_staff` | `staff.yaml` | 18 | DOCUMENT |
| `ask_facility_staff` | `staff.yaml` | 24 | FACILITY |
| `ask_misc_staff` | `staff.yaml` | 16 | MISC |
| `ask_procedure_staff` | `staff.yaml` | 18 | PROCEDURE |
| `ask_technology_staff` | `staff.yaml` | 17 | TECHNOLOGY |
| `ask_statistics` | `statistics.yaml` | 17 |  |
| `ask_organization_statistics` | `statistics.yaml` | 17 | ORGANIZATION |
| `ask_facility_statistics` | `statistics.yaml` | 17 | FACILITY, ORGANIZATION |
| `ask_document_statistics` | `statistics.yaml` | 16 | DOCUMENT, MISC |
| `ask_department_statistics` | `statistics.yaml` | 16 | DEPARTMENT, FACILITY |
| `ask_health_service_statistics` | `statistics.yaml` | 18 | HEALTH_SERVICE, ORGANIZATION |
| `ask_technology_statistics` | `statistics.yaml` | 18 | TECHNOLOGY |
| `ask_misc_statistics` | `statistics.yaml` | 18 | MISC |
| `ask_tax_statistics` | `statistics.yaml` | 16 | TAX |
| `ask_tech_concept_statistics` | `statistics.yaml` | 18 | TECH_CONCEPT |
| `ask_status` | `status.yaml` | 18 |  |
| `ask_document_status` | `status.yaml` | 17 | DOCUMENT, MEDICAL_PROCEDURE, PRODUCT |
| `ask_procedure_status` | `status.yaml` | 16 | PROCEDURE, MISC |
| `ask_health_service_status` | `status.yaml` | 16 | HEALTH_SERVICE |
| `ask_department_status` | `status.yaml` | 18 | DEPARTMENT |
| `ask_misc_status` | `status.yaml` | 18 | MISC |
| `ask_support` | `support.yaml` | 18 |  |
| `ask_health_service_support` | `support.yaml` | 16 | HEALTH_SERVICE |
| `ask_document_support` | `support.yaml` | 17 | DOCUMENT, ORGANIZATION |
| `ask_misc_support` | `support.yaml` | 17 | MISC |
| `ask_department_support` | `support.yaml` | 18 | DEPARTMENT |
| `ask_technology_support` | `support.yaml` | 16 | TECHNOLOGY |
| `ask_procedure_support` | `support.yaml` | 18 | PROCEDURE, ORGANIZATION |
| `ask_tax_collection` | `tax_collection.yaml` | 17 |  |
| `ask_document_tax_collection` | `tax_collection.yaml` | 17 | DOCUMENT, ORGANIZATION, PAYMENT_METHOD |
| `ask_procedure_tax_collection` | `tax_collection.yaml` | 16 | PROCEDURE, DOCUMENT |
| `ask_organization_tax_collection` | `tax_collection.yaml` | 16 | ORGANIZATION, MISC, PROCEDURE |
| `ask_misc_tax_collection` | `tax_collection.yaml` | 16 | MISC |
| `ask_tax_deductions` | `tax_deductions.yaml` | 16 |  |
| `ask_document_tax_deductions` | `tax_deductions.yaml` | 16 | DOCUMENT, MISC |
| `ask_misc_tax_deductions` | `tax_deductions.yaml` | 18 | MISC |
| `ask_payment_method_tax_deductions` | `tax_deductions.yaml` | 18 | PAYMENT_METHOD |
| `ask_role_tax_deductions` | `tax_deductions.yaml` | 18 | ROLE |
| `ask_tax_tax_deductions` | `tax_deductions.yaml` | 16 | TAX |
| `ask_tax_return` | `tax_return.yaml` | 17 |  |
| `ask_document_tax_return` | `tax_return.yaml` | 24 | DOCUMENT, ROLE, ORGANIZATION |
| `ask_misc_tax_return` | `tax_return.yaml` | 18 | MISC |
| `ask_organization_tax_return` | `tax_return.yaml` | 16 | ORGANIZATION, ROLE, DOCUMENT |
| `ask_tax_tax_return` | `tax_return.yaml` | 17 | TAX, ORGANIZATION |
| `ask_tracking` | `tracking.yaml` | 16 |  |
| `ask_document_tracking` | `tracking.yaml` | 17 | DOCUMENT |
| `ask_health_service_tracking` | `tracking.yaml` | 17 | HEALTH_SERVICE |
| `ask_misc_tracking` | `tracking.yaml` | 17 | MISC |
| `ask_procedure_tracking` | `tracking.yaml` | 17 | PROCEDURE |
| `ask_product_tracking` | `tracking.yaml` | 17 | PRODUCT |
| `ask_tech_concept_tracking` | `tracking.yaml` | 16 | TECH_CONCEPT |
| `ask_technology_tracking` | `tracking.yaml` | 16 | TECHNOLOGY |
| `ask_training` | `training.yaml` | 18 |  |
| `ask_technology_training` | `training.yaml` | 17 | TECHNOLOGY |
| `ask_health_service_training` | `training.yaml` | 16 | HEALTH_SERVICE |
| `ask_organization_training` | `training.yaml` | 18 | ORGANIZATION, TECHNOLOGY |
| `ask_role_training` | `training.yaml` | 17 | ROLE, ORGANIZATION |
| `ask_tech_concept_training` | `training.yaml` | 17 | TECH_CONCEPT |
| `ask_department_training` | `training.yaml` | 18 | DEPARTMENT |
| `ask_document_training` | `training.yaml` | 17 | DOCUMENT |
| `ask_misc_training` | `training.yaml` | 18 | MISC |
| `ask_procedure_training` | `training.yaml` | 16 | PROCEDURE |
| `ask_troubleshooting` | `troubleshooting.yaml` | 18 |  |
| `ask_document_troubleshooting` | `troubleshooting.yaml` | 17 | DOCUMENT, ORGANIZATION, MISC |
| `ask_technology_troubleshooting` | `troubleshooting.yaml` | 17 | TECHNOLOGY |
| `ask_health_service_troubleshooting` | `troubleshooting.yaml` | 17 | HEALTH_SERVICE |
| `ask_organization_troubleshooting` | `troubleshooting.yaml` | 17 | ORGANIZATION, DOCUMENT |
| `ask_tech_concept_troubleshooting` | `troubleshooting.yaml` | 17 | TECH_CONCEPT, TECHNOLOGY |
| `ask_waiting_times` | `waiting_times.yaml` | 18 |  |
| `ask_document_waiting_times` | `waiting_times.yaml` | 17 | DOCUMENT, MEDICAL_PROCEDURE |
| `ask_medical_procedure_waiting_times` | `waiting_times.yaml` | 16 | MEDICAL_PROCEDURE, HEALTH_SERVICE |
| `ask_facility_waiting_times` | `waiting_times.yaml` | 18 | FACILITY |
| `ask_procedure_waiting_times` | `waiting_times.yaml` | 16 | PROCEDURE |
| `ask_department_waiting_times` | `waiting_times.yaml` | 18 | DEPARTMENT |
| `ask_health_service_waiting_times` | `waiting_times.yaml` | 17 | HEALTH_SERVICE |
| `ask_misc_waiting_times` | `waiting_times.yaml` | 17 | MISC, DOCUMENT |
| `ask_organization_waiting_times` | `waiting_times.yaml` | 17 | ORGANIZATION, PROCEDURE |
| `ask_warranty` | `warranty.yaml` | 15 |  |
| `ask_document_warranty` | `warranty.yaml` | 18 | DOCUMENT, ORGANIZATION, TAX |
| `ask_product_warranty` | `warranty.yaml` | 18 | PRODUCT |
| `ask_health_service_warranty` | `warranty.yaml` | 16 | HEALTH_SERVICE |
| `ask_technology_warranty` | `warranty.yaml` | 18 | TECHNOLOGY |
| `ask_organization_warranty` | `warranty.yaml` | 17 | ORGANIZATION, ROLE, DOCUMENT |
| `ask_tax_warranty` | `warranty.yaml` | 16 | TAX |
