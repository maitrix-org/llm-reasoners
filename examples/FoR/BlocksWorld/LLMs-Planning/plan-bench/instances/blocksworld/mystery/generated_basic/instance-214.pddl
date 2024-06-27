

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a d)
(planet b)
(planet c)
(planet d)
(province a)
(province b)
(province c)
)
(:goal
(and
(craves b c)
(craves c d)
(craves d a))
)
)


