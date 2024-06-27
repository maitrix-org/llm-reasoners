

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b d)
(craves c b)
(planet d)
(province a)
(province c)
)
(:goal
(and
(craves a c)
(craves b a))
)
)


