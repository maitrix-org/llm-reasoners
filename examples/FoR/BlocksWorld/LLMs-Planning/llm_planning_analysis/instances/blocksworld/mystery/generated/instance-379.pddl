(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects a e h b)
(:init 
(harmony)
(planet a)
(planet e)
(planet h)
(planet b)
(province a)
(province e)
(province h)
(province b)
)
(:goal
(and
(craves a e)
(craves e h)
(craves h b)
)))