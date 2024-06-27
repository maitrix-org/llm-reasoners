

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a c)
(on b d)
(ontable c)
(on d a)
(clear b)
)
(:goal
(and
(on b d)
(on c b)
(on d a))
)
)


