(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects j h e)
(:init 
(harmony)
(planet j)
(planet h)
(planet e)
(province j)
(province h)
(province e)
)
(:goal
(and
(craves j h)
(craves h e)
)))