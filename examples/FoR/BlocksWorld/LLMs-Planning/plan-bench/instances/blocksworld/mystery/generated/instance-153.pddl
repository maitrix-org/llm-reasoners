(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects a h e)
(:init 
(harmony)
(planet a)
(planet h)
(planet e)
(province a)
(province h)
(province e)
)
(:goal
(and
(craves a h)
(craves h e)
)))