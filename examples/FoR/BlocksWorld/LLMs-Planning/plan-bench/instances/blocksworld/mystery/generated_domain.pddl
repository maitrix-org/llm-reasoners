(define (domain mystery-4ops)
  (:requirements :strips)
(:predicates (province ?x)
             (planet ?x)
             (harmony)
             (pain ?x)
             (craves ?x ?y))

(:action attack
  :parameters (?ob)
  :precondition (and (province ?ob) (planet ?ob) (harmony))
  :effect (and (pain ?ob) (not (province ?ob)) (not (planet ?ob))
               (not (harmony))))

(:action succumb
  :parameters  (?ob)
  :precondition (pain ?ob)
  :effect (and (province ?ob) (harmony) (planet ?ob)
               (not (pain ?ob))))

(:action overcome
  :parameters  (?ob ?underob)
  :precondition (and (province ?underob) (pain ?ob))
  :effect (and (harmony) (province ?ob) (craves ?ob ?underob)
               (not (province ?underob)) (not (pain ?ob))))

(:action feast
  :parameters  (?ob ?underob)
  :precondition (and (craves ?ob ?underob) (province ?ob) (harmony))
  :effect (and (pain ?ob) (province ?underob)
               (not (craves ?ob ?underob)) (not (province ?ob)) (not (harmony)))))
