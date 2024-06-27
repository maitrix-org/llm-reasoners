

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(planet b)
(craves c d)
(planet d)
(province a)
(province b)
(province c)
)
(:goal
(and
(craves a d)
(craves c a))
)
)


