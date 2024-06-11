(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects f j e)
(:init 
(harmony)
(planet f)
(planet j)
(planet e)
(province f)
(province j)
(province e)
)
(:goal
(and
(craves f j)
(craves j e)
)))